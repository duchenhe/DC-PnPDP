from collections.abc import Callable

import torch
import tqdm

import algorithms.utils as autils
from algorithms.base import BaseEDMSampler
from algorithms.SH import spectral_homogenization_2d_batched

LinearOp = Callable[[torch.Tensor], torch.Tensor]


class DCPnPDP(BaseEDMSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            raise ValueError("`y`, `A`, and `AT` must be provided for DiffPIRDC sampling.")

        net = self.net
        t_steps = self.get_t_steps(latents)
        print(t_steps, t_steps.shape)

        ATy = AT(y)

        def A_cg(x, rho_tik):
            return AT(A(x)) + rho_tik * x

        def b_cg(x, rho_tik):
            return ATy + rho_tik * x

        Acg_fn = A_cg

        pbar = tqdm.tqdm(range(len(t_steps) - 1), total=len(t_steps) - 1, colour="blue")

        x = torch.zeros_like(ATy)
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)

        v = x_init
        v_prev = v.clone()
        residual_history = {
            "method": "DCPnPDP",
            "definitions": {
                "primal_residual": "||x_k - z_k||_2, where x is the data-consistency variable and z is the prior variable.",
                "dual_residual": "rho_k * ||z_k - z_{k-1}||_2 if rho_k > 0 else ||z_k - z_{k-1}||_2.",
                "dual_residual_raw": "||z_k - z_{k-1}||_2.",
                "data_fidelity_residual": "||A x0_k - y||_2 / ||y||_2, where x0_k is the diffusion prior image.",
                "u_norm": "||u_k||_2, where u accumulates the consensus residual history.",
            },
            "records": [],
        }
        runtime_history = {
            "method": "DCPnPDP",
            "definitions": {
                "outer_iter_time_s": "Wall-clock time of one outer iteration.",
                "denoise_time_s": "Time spent in the diffusion denoiser forward pass.",
                "dc_time_s": "Time spent solving the data-consistency sub-problem.",
                "sh_time_s": "Time spent in spectral homogenization.",
                "other_time_s": "Outer iteration time minus denoise/DC/SH time.",
                "peak_memory_mb": "Peak GPU memory allocated during the outer iteration.",
            },
            "records": [],
        }

        for i in pbar:  # 0, ..., N-1
            t_cur = t_steps[i]
            self._reset_peak_memory_stats(v.device)
            outer_start = self._timer_start(v.device)

            rho_tik = w_tik * (1 / t_cur**2).item()

            # * 1. x update， data sub-problem
            x_in = v - u

            bcg = b_cg(x_in, rho_tik)

            dc_start = self._timer_start(v.device)
            x = autils.cg_uni(Acg_fn, bcg, x_in, rho=rho_tik, maxiter=num_cg)
            dc_time = self._timer_end(dc_start, v.device)
            # *----------------------------------------------------------------------------

            # * 2. v update， prior sub-problem
            x_cur = x + u
            sh_time = 0.0

            # if i > 0:
            if i != len(t_steps) - 1:
                sh_start = self._timer_start(v.device)
                x_cur, info = spectral_homogenization_2d_batched(
                    v=x_cur,
                    x_hat=v,
                    sigma=t_cur,
                    batch_size=50,
                    smooth_ks=7,
                    eps_ratio=1e-6,
                    aggregate="batch_mean",
                )
                sh_time = self._timer_end(sh_start, v.device)

            denoise_start = self._timer_start(v.device)
            v = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16)
            denoise_time = self._timer_end(denoise_start, v.device)

            # *----------------------------------------------------------------------------

            primal_residual = self._l2_norm(x - v)
            dual_residual_raw = self._l2_norm(v - v_prev)
            dual_scale = rho_tik if rho_tik > 0 else 1.0
            _, data_fidelity_residual = self._measurement_residual(v, y, A)

            # * 3. u update
            u = u + (x - v)
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
            v_prev = v.clone()
            outer_time = self._timer_end(outer_start, v.device)
            runtime_history["records"].append(
                {
                    "iteration": i + 1,
                    "sigma": float(t_cur),
                    "outer_iter_time_s": outer_time,
                    "denoise_time_s": denoise_time,
                    "dc_time_s": dc_time,
                    "sh_time_s": sh_time,
                    "other_time_s": max(outer_time - denoise_time - dc_time - sh_time, 0.0),
                    "peak_memory_mb": self._get_peak_memory_mb(v.device),
                }
            )
            # *----------------------------------------------------------------------------

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

        self._save_residual_history(residual_history)
        self._save_runtime_profile(runtime_history)
        return v
