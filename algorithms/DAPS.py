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
        u = torch.zeros_like(x_init)
        x_corr_prev = None
        residual_history = {
            "method": "DAPS",
            "definitions": {
                "primal_residual": "Predictor-corrector discrepancy ||x_corr_k - x_pred_k||_2, where x_pred is the PF-ODE prediction and x_corr is the Langevin-corrected sample.",
                "dual_residual": "Surrogate dual residual ||x_corr_k - x_corr_{k-1}||_2.",
                "dual_residual_raw": "Surrogate raw dual residual ||x_corr_k - x_corr_{k-1}||_2.",
                "data_fidelity_residual": "||A x_corr_k - y||_2 / ||y||_2, where x_corr_k is the Langevin-corrected sample.",
                "u_norm": "||u_k||_2, where u accumulates x_corr_k - x_pred_k but does not affect the updates.",
            },
            "records": [],
        }
        runtime_history = {
            "method": "DAPS",
            "definitions": {
                "outer_iter_time_s": "Wall-clock time of one outer iteration.",
                "denoise_time_s": "Time spent in the PF-ODE predictor denoising passes.",
                "dc_time_s": "Time spent in the Langevin corrector.",
                "sh_time_s": "Always zero for DAPS.",
                "other_time_s": "Outer iteration time minus denoise/DC/SH time.",
                "peak_memory_mb": "Peak GPU memory allocated during the outer iteration.",
            },
            "records": [],
        }

        pbar = tqdm.tqdm(range(len(t_steps) - 1), total=len(t_steps) - 1, colour="blue")

        for i in pbar:
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            x_cur = x_next
            self._reset_peak_memory_stats(x_cur.device)
            outer_start = self._timer_start(x_cur.device)

            pred_start = self._timer_start(x_cur.device)
            x0hat = self._predict_x0_with_ode(x_cur, t_cur, class_labels)
            denoise_time = self._timer_end(pred_start, x_cur.device)
            corr_start = self._timer_start(x_cur.device)
            x0y = self.lgvd.sample(
                x0hat=x0hat,
                operator=A,
                measurement=y,
                sigma=t_cur,
                ratio=i / max(len(t_steps) - 1, 1),
                num_steps=self.lgvd_num_steps,
                verbose=False,
            )
            dc_time = self._timer_end(corr_start, x_cur.device)

            primal_residual = self._l2_norm(x0y - x0hat)
            if x_corr_prev is None:
                dual_residual_raw = 0.0
            else:
                dual_residual_raw = self._l2_norm(x0y - x_corr_prev)
            _, data_fidelity_residual = self._measurement_residual(x0y, y, A)
            u_update = x0y - x0hat
            u_update = torch.where(torch.isfinite(u_update), u_update, torch.zeros_like(u_update))
            u = u + u_update
            residual_history["records"].append(
                {
                    "iteration": i + 1,
                    "sigma": float(t_cur),
                    "rho_tik": 0.0,
                    "primal_residual": primal_residual,
                    "dual_residual": dual_residual_raw,
                    "dual_residual_raw": dual_residual_raw,
                    "data_fidelity_residual": data_fidelity_residual,
                    "u_norm": self._l2_norm(u),
                }
            )
            x_corr_prev = x0y.detach().clone()

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

            outer_time = self._timer_end(outer_start, x_cur.device)
            runtime_history["records"].append(
                {
                    "iteration": i + 1,
                    "sigma": float(t_cur),
                    "outer_iter_time_s": outer_time,
                    "denoise_time_s": denoise_time,
                    "dc_time_s": dc_time,
                    "sh_time_s": 0.0,
                    "other_time_s": max(outer_time - denoise_time - dc_time, 0.0),
                    "peak_memory_mb": self._get_peak_memory_mb(x_cur.device),
                }
            )
            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

        self._save_residual_history(residual_history)
        self._save_runtime_profile(runtime_history)
        self.save_intermediate_results(x_t_list, d_cur_list, denoised_list)
        return x_next
