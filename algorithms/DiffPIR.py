import math

import torch
import tqdm

from utils.result import save_nii_image
from algorithms.base import BaseEDMSampler
import algorithms.utils as autils


class DiffPIR(BaseEDMSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(
        self,
        latents,
        x_init,
        class_labels=None,
        y=None,
        A=None,
        AT=None,
        num_cg=10,
        w_tik=0.0,
    ):
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

            # ---- Denoising + Correction ----
            denoised = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16)

            # *----------------------------------------------------------------------------
            # * Data Consistency via Conjugate Gradient
            denoised_temp = denoised.movedim(0, 1)  # (300,1,256,256) -> (1,300,256,256) # (N,C,H,W) -> (C,N,H,W)

            rho_tik = w_tik * (1 / t_cur**2).item()
            bcg = ATy + rho_tik * denoised_temp

            x_0_t_hat = autils.cg_uni(Acg_fn, bcg, denoised_temp, rho=rho_tik, maxiter=num_cg)

            x_0_t_hat = x_0_t_hat.movedim(0, 1)  # (1,300,256,256) -> (300,1,256,256)

            # *----------------------------------------------------------------------------

            # ---- EDM Euler Update ----
            d_cur = (x_cur - x_0_t_hat) / t_cur  # score

            # ---- Re-Nosing ----
            noises = torch.randn_like(x_0_t_hat)

            eta = 1

            sigma_sto = eta * t_next
            sigma_det = math.sqrt(1 - eta**2) * t_next

            noise_sto = sigma_sto * noises
            noise_det = sigma_det * (t_next) * d_cur

            x_next = x_0_t_hat + noise_det + noise_sto

        return x_next
