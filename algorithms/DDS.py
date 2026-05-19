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

        pbar = tqdm.tqdm(range(len(t_steps) - 1), total=len(t_steps) - 1, colour="blue")
        for i in pbar:
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            x_cur = x_next

            denoised = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16).to(torch.float32)

            rho_tik = w_tik * (1 / max(float(t_cur), 1e-8) ** 2)
            bcg = ATy + rho_tik * denoised
            x0_hat = autils.cg_uni(A_cg, bcg, denoised, rho=rho_tik, maxiter=num_cg).to(torch.float32)

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

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

        self.save_intermediate_results(x_t_list, d_cur_list, denoised_list)
        return x_next
