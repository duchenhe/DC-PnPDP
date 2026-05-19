from collections.abc import Callable

import torch
import tqdm

from algorithms.base import BaseEDMSampler
from utils.result import save_nii_image

LinearOp = Callable[[torch.Tensor], torch.Tensor]


class LangevinDynamics:
    """Langevin corrector following the official DAPS implementation."""

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio

    def get_lr(self, ratio):
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr

    def _measurement_gradient(self, x, operator, measurement):
        if hasattr(operator, "gradient") and callable(operator.gradient):
            return operator.gradient(x, measurement)

        residual = operator(x) - measurement
        loss = residual.abs().square().flatten(start_dim=1).sum(dim=1).sum()
        return torch.autograd.grad(loss, x)[0]

    def sample(self, x0hat, operator, measurement, sigma, ratio, num_steps=None, verbose=False):
        steps = max(int(self.num_steps if num_steps is None else num_steps), 1)
        lr = self.get_lr(ratio)
        sigma_sq = torch.as_tensor(float(sigma) ** 2, device=x0hat.device, dtype=x0hat.dtype).clamp_min(1e-12)
        noise_scale = torch.sqrt(torch.as_tensor(2.0 * lr, device=x0hat.device, dtype=x0hat.dtype))
        x0hat = x0hat.detach()

        with torch.enable_grad():
            x = x0hat.clone().detach().requires_grad_(True)
            optimizer = torch.optim.SGD([x], lr)
            pbar = tqdm.trange(steps) if verbose else range(steps)

            for _ in pbar:
                optimizer.zero_grad(set_to_none=True)
                gradient = self._measurement_gradient(x, operator, measurement) / (2 * self.tau**2)
                gradient = gradient + (x - x0hat) / sigma_sq
                x.grad = gradient.detach()
                del gradient

                optimizer.step()

                with torch.no_grad():
                    x.add_(noise_scale * torch.randn_like(x))

                if torch.isnan(x).any():
                    # print("NaN detected in Langevin dynamics, stopping early.")
                    return torch.zeros_like(x0hat)

        return x.detach()


class DAPS(BaseEDMSampler):
    """DAPS with official-style PF-ODE predictor and Langevin corrector."""

    def __init__(
        self,
        *args,
        diffusion_num_steps=5,
        diffusion_sigma_min=0.01,
        lgvd_num_steps=20,
        lgvd_lr=5e-6,
        lgvd_tau=0.25,
        lgvd_lr_min_ratio=0.01,
        denoise_batch_size=16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.diffusion_num_steps = diffusion_num_steps
        self.diffusion_sigma_min = diffusion_sigma_min
        self.lgvd_num_steps = lgvd_num_steps
        self.denoise_batch_size = denoise_batch_size
        self.lgvd = LangevinDynamics(
            num_steps=lgvd_num_steps,
            lr=lgvd_lr,
            tau=lgvd_tau,
            lr_min_ratio=lgvd_lr_min_ratio,
        )

    def _get_local_t_steps(self, ref_tensor, sigma_max, num_steps):
        sigma_min = max(float(self.diffusion_sigma_min), float(self.net.sigma_min))
        sigma_start = max(float(sigma_max), sigma_min)
        inner_steps = max(int(num_steps), 1)

        if inner_steps == 1 or sigma_start <= sigma_min * (1 + 1e-12):
            return torch.tensor([sigma_start, 0.0], dtype=torch.float64, device=ref_tensor.device)

        step_indices = torch.arange(inner_steps, dtype=torch.float64, device=ref_tensor.device)
        t_steps = (
            sigma_start ** (1 / self.rho)
            + step_indices / (inner_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_start ** (1 / self.rho))
        ) ** self.rho
        return torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    def _predict_x0_with_ode(self, x_start, sigma_max, class_labels=None):
        local_t_steps = self._get_local_t_steps(x_start, sigma_max, self.diffusion_num_steps)
        x_next = x_start.to(torch.float32)

        with torch.no_grad():
            for i in range(len(local_t_steps) - 1):
                t_cur = local_t_steps[i].to(torch.float32)
                t_next = local_t_steps[i + 1].to(torch.float32)

                denoised = self._denoise_batchwise(
                    self.net,
                    x_next,
                    t_cur,
                    class_labels,
                    batch_size=self.denoise_batch_size,
                )
                d_cur = (x_next - denoised) / t_cur
                x_next = x_next + (t_next - t_cur) * d_cur

        return x_next.detach()

    def sample(
        self,
        latents,
        x_init,
        class_labels=None,
        randn_like=torch.randn_like,
        y: torch.Tensor | None = None,
        A: LinearOp | None = None,
        AT: LinearOp | None = None,
        num_cg=10,
        w_tik=0.0,
    ):
        del AT, num_cg, w_tik  # kept for entrypoint compatibility

        if y is None or A is None:
            raise ValueError("`y` and `A` must be provided for DAPS sampling.")

        t_steps = self.get_t_steps(latents).to(torch.float32)
        x_next = x_init.to(torch.float32) + latents.to(torch.float32) * t_steps[0]

        if self.save_path:
            save_nii_image(x_next, f"{self.save_path}/x_init.nii.gz")

        x_t_list = []
        d_cur_list = []
        denoised_list = []

        pbar = tqdm.tqdm(range(len(t_steps) - 1), total=len(t_steps) - 1, colour="blue")

        for i in pbar:
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            x_cur = x_next

            x0hat = self._predict_x0_with_ode(x_cur, t_cur, class_labels)
            x0y = self.lgvd.sample(
                x0hat=x0hat,
                operator=A,
                measurement=y,
                sigma=t_cur,
                ratio=i / max(len(t_steps) - 1, 1),
                num_steps=self.lgvd_num_steps,
                verbose=False,
            )

            if self.save_intermediates:
                x_t_list.append(x_cur.detach().cpu().numpy())
                d_cur_list.append((x_cur - x0hat).detach().cpu().numpy())
                denoised_list.append(x0y.detach().cpu().numpy())

            if float(t_next) > 0:
                if str(self.noise_control).lower() in {"deterministic", "zero"}:
                    noise = torch.zeros_like(x0y)
                else:
                    noise = randn_like(x0y)
                x_next = x0y + t_next * noise
            else:
                x_next = x0y

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

        self.save_intermediate_results(x_t_list, d_cur_list, denoised_list)
        return x_next
