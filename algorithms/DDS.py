import math
from collections.abc import Callable

import torch
import tqdm

import algorithms.utils as autils
from algorithms.base import BaseEDMSampler
from utils.result import save_nii_image

LinearOp = Callable[[torch.Tensor], torch.Tensor]


class DDS(BaseEDMSampler):
    def __init__(self, *args, eta: float = 0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = float(eta)

    def sample(
        self,
        latents,
        x_init,
        class_labels=None,
        y: torch.Tensor | None = None,
        A: LinearOp | None = None,
        AT: LinearOp | None = None,
        num_cg=10,
        w_tik=0.0,
    ):
        if y is None or A is None or AT is None:
            raise ValueError("`y`, `A`, and `AT` must be provided for DDS sampling.")
        
        num_cg = 5  # 使用固定的 CG 迭代次数

        net = self.net
        t_steps = self.get_t_steps(latents).to(torch.float32)

        x_next = x_init.to(torch.float32) + latents.to(torch.float32) * t_steps[0]
        if self.save_path:
            save_nii_image(x_next, f"{self.save_path}/x_init.nii.gz")

        ATy = AT(y).to(torch.float32)

        def A_cg(x, rho_tik):
            return AT(A(x)) + rho_tik * x

        x_t_list = []
        d_cur_list = []
        denoised_list = []
        u = torch.zeros_like(x_init)
        x_dc_prev = None
        residual_history = {
            "method": "DDS",
            "definitions": {
                "primal_residual": "Surrogate consensus residual ||x_dc_k - x_prior_k||_2, where x_prior is the denoised sample before CG and x_dc is the CG-corrected sample.",
                "dual_residual": "Surrogate dual residual rho_k * ||x_dc_k - x_dc_{k-1}||_2 if rho_k > 0 else ||x_dc_k - x_dc_{k-1}||_2.",
                "dual_residual_raw": "Surrogate raw dual residual ||x_dc_k - x_dc_{k-1}||_2.",
                "data_fidelity_residual": "||A x_prior_k - y||_2 / ||y||_2, where x_prior_k is the denoised sample before CG.",
                "u_norm": "||u_k||_2, where u accumulates x_dc_k - x_prior_k but does not affect the sub-problem updates.",
            },
            "records": [],
        }
        runtime_history = {
            "method": "DDS",
            "definitions": {
                "outer_iter_time_s": "Wall-clock time of one outer iteration.",
                "denoise_time_s": "Time spent in the diffusion denoiser forward pass.",
                "dc_time_s": "Time spent solving the data-consistency sub-problem.",
                "sh_time_s": "Always zero for DDS.",
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

            denoise_start = self._timer_start(x_cur.device)
            denoised = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16).to(torch.float32)
            denoise_time = self._timer_end(denoise_start, x_cur.device)

            rho_tik = w_tik * (1 / max(float(t_cur), 1e-8) ** 2)
            bcg = ATy + rho_tik * denoised
            dc_start = self._timer_start(x_cur.device)
            x0_hat = autils.cg_uni(A_cg, bcg, denoised, rho=rho_tik, maxiter=num_cg).to(torch.float32)
            dc_time = self._timer_end(dc_start, x_cur.device)

            primal_residual = self._l2_norm(x0_hat - denoised)
            if x_dc_prev is None:
                dual_residual_raw = 0.0
            else:
                dual_residual_raw = self._l2_norm(x0_hat - x_dc_prev)
            dual_scale = rho_tik if rho_tik > 0 else 1.0
            _, data_fidelity_residual = self._measurement_residual(denoised, y, A)
            u = u + (x0_hat - denoised)
            residual_history["records"].append(
                {
                    "iteration": i + 1,
                    "sigma": float(t_cur),
                    "rho_tik": float(rho_tik),
                    "primal_residual": primal_residual,
                    "dual_residual": dual_scale * dual_residual_raw,
                    "dual_residual_raw": dual_residual_raw,
                    "data_fidelity_residual": data_fidelity_residual,
                    "u_norm": self._l2_norm(u),
                }
            )
            x_dc_prev = x0_hat.detach().clone()

            d_cur = (x_cur - x0_hat) / t_cur

            if self.save_intermediates:
                x_t_list.append(x_cur.detach().cpu().numpy())
                d_cur_list.append(d_cur.detach().cpu().numpy())
                denoised_list.append(x0_hat.detach().cpu().numpy())

            if float(t_next) > 0:
                noise = torch.randn_like(x0_hat)
                sigma_sto = self.eta * t_next
                sigma_det = math.sqrt(max(1.0 - self.eta**2, 0.0)) * t_next
                x_next = x0_hat + sigma_det * d_cur + sigma_sto * noise
            else:
                x_next = x0_hat

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
