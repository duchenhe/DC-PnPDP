import math
from collections.abc import Callable

import torch
import tqdm

from utils.result import save_nii_image
from algorithms.base import BaseEDMSampler
import algorithms.utils as autils

LinearOp = Callable[[torch.Tensor], torch.Tensor]


class DiffPIR(BaseEDMSampler):
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
            raise ValueError("`y`, `A`, and `AT` must be provided for DiffPIR sampling.")

        net = self.net
        t_steps = self.get_t_steps(latents)

        x_next = x_init.to(torch.float32) + latents.to(torch.float32) * t_steps[0]
        if self.save_path:
            save_nii_image(x_next, f"{self.save_path}/x_init.nii.gz")

        def A_cg(x, rho_tik):
            return AT(A(x)) + rho_tik * x

        Acg_fn = A_cg
        ATy = AT(y)

        pbar = tqdm.tqdm(range(len(t_steps) - 1), total=len(t_steps) - 1, colour="blue")

        # for i, (t_cur, t_next) in pbar:
        for i in pbar:  # 0, ..., N-1
            t_cur, t_next = t_steps[i], t_steps[i + 1]

            x_cur = x_next

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

            # ---- Denoising ----
            x_0_cur = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16)

            # *----------------------------------------------------------------------------
            # * Data Consistency via Conjugate Gradient
            rho_tik = w_tik * (1 / t_cur**2).item()
            bcg = ATy + rho_tik * x_0_cur

            x_0_cur_hat = autils.cg_uni(Acg_fn, bcg, x_0_cur, rho=rho_tik, maxiter=num_cg)
            # *----------------------------------------------------------------------------

            # ---- EDM Euler Update ----
            d_cur = (x_cur - x_0_cur_hat) / t_cur  # score

            # ---- Re-Nosing ----
            noises = torch.randn_like(x_0_cur_hat)

            eta = 1

            sigma_sto = eta * t_next
            sigma_det = math.sqrt(1 - eta**2) * t_next

            noise_sto = sigma_sto * noises
            noise_det = sigma_det * (t_next) * d_cur

            x_next = x_0_cur_hat + noise_det + noise_sto

        return x_next
