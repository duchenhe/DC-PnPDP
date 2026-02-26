import torch
import torch.nn.functional as F


@torch.no_grad()
def spectral_homogenization_2d(
    v: torch.Tensor,  # [B,C,H,W], real-valued (or complex, see notes)
    x_hat: torch.Tensor,  # [B,C,H,W], proxy "clean" image, e.g., z_prev
    sigma: float | torch.Tensor,  # scalar noise std in the same scale as v
    smooth_ks: int = 7,  # frequency-domain smoothing kernel (odd)
    eps_ratio: float = 1e-6,  # floor as a ratio of P_white
    aggregate: str = "batch_mean",  # "none" | "batch_mean" | "batch_median" (simple)
):
    """
    Minimal FIRE-style complementary colored renoising (spectral-diagonal approximation).

    Returns:
        v_renoised: v + xi
        info: dict with diagnostics (optional)
    """
    assert v.ndim == 4, "Expect v in [B,C,H,W]"
    B, C, H, W = v.shape
    device = v.device
    dtype = v.dtype

    # ---- 1) Estimate colored error r = v - x_hat ----
    r = v - x_hat

    # ---- 2) FFT and estimate power spectrum P_r = |FFT(r)|^2 ----
    # Works for real tensors; fft2 will return complex.
    R = torch.fft.fft2(r, dim=(-2, -1))
    P_r = R.real**2 + R.imag**2  # [B,C,H,W], real-valued power

    # Optional: aggregate across batch/channels to stabilize PSD estimate.
    # This makes the renoising "shared" across batch (often more stable).
    if aggregate == "batch_mean":
        P_r_est = P_r.mean(dim=(0, 1), keepdim=True)  # [1,1,H,W]
        P_r_est = P_r_est.expand(B, C, H, W)
    elif aggregate == "batch_median":
        # median over B*C samples per frequency (more robust; slightly slower)
        P_r_flat = P_r.reshape(B * C, H, W)
        P_r_med = P_r_flat.median(dim=0).values  # [H,W]
        P_r_est = P_r_med.view(1, 1, H, W).expand(B, C, H, W)
    else:
        P_r_est = P_r

    # ---- 3) Smooth the power spectrum in frequency domain (stabilizes complement) ----
    if smooth_ks is not None and smooth_ks > 1:
        assert smooth_ks % 2 == 1, "smooth_ks must be odd"
        pad = smooth_ks // 2
        # avg_pool2d expects [N,C,H,W]; treat (B*C) as batch, 1 as channel
        P_tmp = P_r_est.reshape(B * C, 1, H, W)
        P_tmp = F.avg_pool2d(P_tmp, kernel_size=smooth_ks, stride=1, padding=pad)
        P_r_est = P_tmp.reshape(B, C, H, W)

    # ---- 4) Build complementary PSD: P_xi = max(eps, P_white - P_r_est) ----
    # Target white-noise power per frequency under unnormalized FFT:
    #   if x ~ N(0, sigma^2) i.i.d., then E|FFT(x)[w]|^2 = (H*W)*sigma^2
    if not torch.is_tensor(sigma):
        sigma_t = torch.tensor(float(sigma), device=device, dtype=torch.float32)
    else:
        sigma_t = sigma.to(device=device).float()

    N = float(H * W)
    P_white = (sigma_t**2) * N  # scalar in float32
    eps = (eps_ratio * P_white).clamp_min(0.0)

    # Broadcast P_white/eps to [B,C,H,W]
    P_white_bc = P_white.view(1, 1, 1, 1).expand(B, C, H, W)
    eps_bc = eps.view(1, 1, 1, 1).expand(B, C, H, W)

    P_xi = (P_white_bc - P_r_est).clamp_min(0.0)
    P_xi = torch.maximum(P_xi, eps_bc)

    # ---- 5) Sample xi with the desired power spectrum ----
    # Strategy:
    #   n ~ N(0,1) in spatial domain => E|FFT(n)|^2 = (H*W)
    #   Xi_f = sqrt(P_xi / (H*W)) * FFT(n) => E|Xi_f|^2 = P_xi
    n = torch.randn((B, C, H, W), device=device, dtype=torch.float32)
    Nf = torch.fft.fft2(n, dim=(-2, -1))  # complex64/128

    scale = torch.sqrt(P_xi / N).to(Nf.dtype)  # real -> complex dtype
    Xi_f = scale * Nf
    xi = torch.fft.ifft2(Xi_f, dim=(-2, -1)).real  # should be ~real due to symmetry

    # Cast back to v dtype if needed
    xi = xi.to(dtype=dtype)
    v_renoised = v + xi

    info = {
        "sigma": float(sigma_t.item()),
        "P_white": float(P_white.item()),
        "xi_std": float(xi.float().std(unbiased=False).item()),
        "r_std": float(r.float().std(unbiased=False).item()),
        "P_r_mean": float(P_r_est.float().mean().item()),
        "P_xi_mean": float(P_xi.float().mean().item()),
    }
    return v_renoised, info


@torch.no_grad()
def spectral_homogenization_2d_batched(
    v: torch.Tensor,  # [B,C,H,W], real-valued tensor
    x_hat: torch.Tensor,  # [B,C,H,W], proxy "clean" image, e.g., z_prev
    sigma: float | torch.Tensor,  # scalar noise std, same scale as v
    batch_size: int = 50,  # samples per batch to control VRAM usage
    smooth_ks: int = 7,  # frequency-domain smoothing kernel size (odd)
    eps_ratio: float = 1e-6,  # minimum PSD floor (relative to white-noise power)
    aggregate: str = "batch_mean",  # "none" | "batch_mean" | "batch_median"
    verbose: bool = False,  # whether to print progress
):
    """
    Batched version of complementary colored renoising (FIRE-style).

    This function splits large batches into smaller chunks to avoid OOM.
    It is especially useful for large-scale data (e.g., 500x256x256).

    Algorithm:
    1. Estimate colored error r = v - x_hat
    2. Estimate the error power spectrum P_r via FFT
    3. Compute complementary PSD P_xi = max(eps, P_white - P_r)
    4. Generate colored noise xi using the complementary PSD
    5. Return v_renoised = v + xi

    Args:
        v:  [B,C,H,W]
        x_hat: proxy clean image [B,C,H,W]
        sigma: target noise standard deviation
        batch_size: samples per batch (smaller reduces VRAM usage)
        smooth_ks: frequency-domain smoothing kernel size for PSD stabilization
        eps_ratio: lower-bound ratio for the PSD
        aggregate: PSD aggregation mode
            - "none": no aggregation, each sample independent
            - "batch_mean": batch mean (more stable, recommended)
            - "batch_median": batch median (more robust, slightly slower)
        verbose: whether to show progress

    Returns:
        v_renoised: renoised tensor [B,C,H,W]
        info: dict with diagnostics

    Example:
        >>> v = torch.randn(500, 1, 256, 256, device='cuda')
        >>> x_hat = v + torch.randn_like(v) * 0.1
        >>> v_renoised, info = complementary_colored_renoise_2d_batched(
        ...     v, x_hat, sigma=0.5, batch_size=50
        ... )
        >>> print(f"Processed {v.shape[0]} samples in {v.shape[0]//50} batches")
    """
    assert v.ndim == 4, "Expected input tensor shape [B,C,H,W]"
    assert v.shape == x_hat.shape, "v and x_hat must have the same shape"

    B, C, H, W = v.shape
    device = v.device
    dtype = v.dtype

    # If the batch size is >= total size, fall back to the base function.
    if batch_size >= B:
        if verbose:
            print(f"Batch size ({batch_size}) >= total samples ({B}); processing directly")
        return spectral_homogenization_2d(v, x_hat, sigma, smooth_ks, eps_ratio, aggregate)

    # Compute number of batches.
    num_batches = (B + batch_size - 1) // batch_size

    if verbose:
        print(f"Split {B} samples into {num_batches} batches, up to {batch_size} each")

    # ==== Step 1: Global PSD estimate (if using aggregation) ====
    P_r_global = None
    if aggregate in ["batch_mean", "batch_median"]:
        if verbose:
            print("Computing global PSD estimate...")

        # Compute batch PSDs, then aggregate.
        P_r_list = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, B)

            v_batch = v[start_idx:end_idx]
            x_hat_batch = x_hat[start_idx:end_idx]

            # Compute error and PSD.
            r_batch = v_batch - x_hat_batch
            R_batch = torch.fft.fft2(r_batch, dim=(-2, -1))
            P_r_batch = R_batch.real**2 + R_batch.imag**2  # [batch_size, C, H, W]

            P_r_list.append(P_r_batch)

        # Merge PSDs from all batches.
        P_r_all = torch.cat(P_r_list, dim=0)  # [B, C, H, W]

        # Aggregate according to the chosen mode.
        if aggregate == "batch_mean":
            P_r_global = P_r_all.mean(dim=(0, 1), keepdim=True)  # [1, 1, H, W]
        elif aggregate == "batch_median":
            P_r_flat = P_r_all.reshape(B * C, H, W)
            P_r_global = P_r_flat.median(dim=0).values  # [H, W]
            P_r_global = P_r_global.view(1, 1, H, W)

        # Release intermediates.
        del P_r_list, P_r_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if verbose:
            print(f"Global PSD estimate done, shape: {P_r_global.shape}")

    # ==== Step 2: Batched renoising ====
    v_renoised_list = []
    info_list = []

    # Progress (optional).
    if verbose:
        try:
            from tqdm import tqdm

            batch_iterator = tqdm(range(num_batches), desc="Colored renoising")
        except ImportError:
            batch_iterator = range(num_batches)
            print("Starting batched processing...")
    else:
        batch_iterator = range(num_batches)

    for i in batch_iterator:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, B)
        current_batch_size = end_idx - start_idx

        # Extract current batch.
        v_batch = v[start_idx:end_idx]
        x_hat_batch = x_hat[start_idx:end_idx]

        # ---- 1) Estimate colored error r = v - x_hat ----
        r_batch = v_batch - x_hat_batch

        # ---- 2) FFT and estimate PSD P_r = |FFT(r)|^2 ----
        R_batch = torch.fft.fft2(r_batch, dim=(-2, -1))
        P_r_batch = R_batch.real**2 + R_batch.imag**2  # [current_batch_size, C, H, W]

        # Use global or local PSD.
        if P_r_global is not None:
            P_r_est = P_r_global.expand(current_batch_size, C, H, W)
        elif aggregate == "none":
            P_r_est = P_r_batch
        else:
            # Local aggregation within the current batch.
            if aggregate == "batch_mean":
                P_r_est = P_r_batch.mean(dim=(0, 1), keepdim=True)
                P_r_est = P_r_est.expand(current_batch_size, C, H, W)
            else:  # batch_median
                P_r_flat = P_r_batch.reshape(current_batch_size * C, H, W)
                P_r_med = P_r_flat.median(dim=0).values
                P_r_est = P_r_med.view(1, 1, H, W).expand(current_batch_size, C, H, W)

        # ---- 3) Smooth PSD in frequency domain (stabilizes complement) ----
        if smooth_ks is not None and smooth_ks > 1:
            assert smooth_ks % 2 == 1, "smooth_ks must be odd"
            pad = smooth_ks // 2
            # avg_pool2d expects [N,C,H,W]; treat (batch*C) as batch, 1 as channel.
            P_tmp = P_r_est.reshape(current_batch_size * C, 1, H, W)
            P_tmp = F.avg_pool2d(P_tmp, kernel_size=smooth_ks, stride=1, padding=pad)
            P_r_est = P_tmp.reshape(current_batch_size, C, H, W)

        # ---- 4) Build complementary PSD: P_xi = max(eps, P_white - P_r_est) ----
        # Target white-noise power under unnormalized FFT:
        #   if x ~ N(0, sigma^2) i.i.d., then E|FFT(x)[w]|^2 = (H*W)*sigma^2
        if not torch.is_tensor(sigma):
            sigma_t = torch.tensor(float(sigma), device=device, dtype=torch.float32)
        else:
            sigma_t = sigma.to(device=device).float()

        N = float(H * W)
        P_white = (sigma_t**2) * N  # float32 scalar
        eps = (eps_ratio * P_white).clamp_min(0.0)

        # Broadcast P_white/eps to [current_batch_size, C, H, W]
        P_white_bc = P_white.view(1, 1, 1, 1).expand(current_batch_size, C, H, W)
        eps_bc = eps.view(1, 1, 1, 1).expand(current_batch_size, C, H, W)

        P_xi = (P_white_bc - P_r_est).clamp_min(0.0)
        P_xi = torch.maximum(P_xi, eps_bc)

        # ---- 5) Sample xi with the desired PSD ----
        # Strategy:
        #   n ~ N(0,1) in spatial domain => E|FFT(n)|^2 = (H*W)
        #   Xi_f = sqrt(P_xi / (H*W)) * FFT(n) => E|Xi_f|^2 = P_xi
        n = torch.randn((current_batch_size, C, H, W), device=device, dtype=torch.float32)
        Nf = torch.fft.fft2(n, dim=(-2, -1))  # complex64/128

        scale = torch.sqrt(P_xi / N).to(Nf.dtype)  # real -> complex dtype
        Xi_f = scale * Nf
        xi = torch.fft.ifft2(Xi_f, dim=(-2, -1)).real  # ~real due to symmetry

        # Cast back to v dtype if needed.
        xi = xi.to(dtype=dtype)
        v_batch_renoised = v_batch + xi

        # Save results.
        v_renoised_list.append(v_batch_renoised)

        # Collect diagnostics (only from the first batch).
        if i == 0:
            batch_info = {
                "sigma": float(sigma_t.item()),
                "P_white": float(P_white.item()),
                "xi_std": float(xi.float().std(unbiased=False).item()),
                "r_std": float(r_batch.float().std(unbiased=False).item()),
                "P_r_mean": float(P_r_est.float().mean().item()),
                "P_xi_mean": float(P_xi.float().mean().item()),
            }
            info_list.append(batch_info)

        # Cleanup intermediates to save VRAM.
        del r_batch, R_batch, P_r_batch, P_r_est, P_xi, n, Nf, Xi_f, xi
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ==== Step 3: Merge all batches ====
    v_renoised = torch.cat(v_renoised_list, dim=0)

    # Use the first batch diagnostics (representative).
    info = info_list[0] if info_list else {}
    info["num_batches"] = num_batches
    info["batch_size"] = batch_size
    info["total_samples"] = B

    if verbose:
        print(f"\nDone! Processed {B} samples in {num_batches} batches")
        print(f"Output shape: {v_renoised.shape}")

    return v_renoised, info
