import numpy as np
import torch
from utils.result import save_nii_image
import tqdm
import algorithms.utils as autils


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


# ==============================
# Base class: common EDM sampling logic
# ==============================
class BaseEDMSampler:
    def __init__(
        self,
        net,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        save_path=None,
        save_intermediates=False,
        noise_control=None,
    ):
        self.net = net
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.save_path = save_path
        self.save_intermediates = save_intermediates
        self.noise_control = noise_control

    def get_t_steps(self, latents):
        """Define the time steps for the EDM schedule."""
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        return t_steps

    def save_intermediate_results(self, x_t_list, d_cur_list, denoised_list):
        """Save intermediate results (common helper)."""
        if self.save_intermediates and self.save_path is not None:
            x_t_list = np.array(x_t_list)
            d_cur_list = np.array(d_cur_list)
            denoised_list = np.array(denoised_list)
            save_nii_image(x_t_list, f"{self.save_path}/x_t_list.nii.gz")
            save_nii_image(d_cur_list, f"{self.save_path}/d_cur_list.nii.gz")
            save_nii_image(denoised_list, f"{self.save_path}/denoised_list.nii.gz")

    def _denoise_batchwise(self, net, x, t_cur, class_labels, batch_size=None):
        """Denoise in batches to avoid GPU out-of-memory.

        Args:
            batch_size: If None, automatically choose a suitable batch size.
        """
        if batch_size is None:
            batch_size = self._estimate_batch_size(x)

        batches = autils.batchfy(x, batch_size)
        with torch.no_grad():
            denoised_batches = [net(xb, t_cur, class_labels).to(torch.float32) for xb in batches]
        return torch.cat(denoised_batches, dim=0)

    def _estimate_batch_size(self, x):
        """Estimate a suitable batch size from available GPU memory and input size."""
        if not torch.cuda.is_available():
            return min(25, x.shape[0])

        try:
            # Get the available GPU memory (bytes).
            free_memory, total_memory = torch.cuda.mem_get_info()

            # Estimate per-sample memory usage (conservative).
            # x is typically shaped as (N, C, H, W).
            single_sample_size = x[0:1].element_size() * x[0:1].nelement()

            # A forward pass typically needs ~5-10x the input memory (incl. activations).
            estimated_memory_per_sample = single_sample_size * 10

            # Keep 30% of memory as a safety buffer.
            usable_memory = free_memory * 0.7

            # Compute batch size.
            estimated_batch_size = int(usable_memory / estimated_memory_per_sample)

            # Clamp to a reasonable range.
            batch_size = max(1, min(estimated_batch_size, x.shape[0], 50))

            return batch_size

        except Exception:
            # If estimation fails, return a conservative default.
            return min(25, x.shape[0])

    def sample(
        self,
        latents,
        x_init,
    ):
        net = self.net
        t_steps = self.get_t_steps(latents)

        # Main sampling loop.
        # x_next = latents.to(torch.float64) * t_steps[0]
        x_next = x_init.to(torch.float64) + latents.to(torch.float64) * t_steps[0]

        save_nii_image(x_next, f"{self.save_path}/x_init.nii.gz") if self.save_path else None

        for i, (t_cur, t_next) in tqdm.tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=len(t_steps) - 1
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels=None).to(torch.float64)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = net(x_next, t_next, class_labels=None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
