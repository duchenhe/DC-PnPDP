from collections.abc import Callable

import torch
import tqdm
import math

from algorithms.base import BaseEDMSampler
from utils.result import save_nii_image

LinearOp = Callable[[torch.Tensor], torch.Tensor]


class DDNM(BaseEDMSampler):
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
        del num_cg, w_tik  # kept for a shared entrypoint signature

        if y is None or A is None or AT is None:
            raise ValueError("`y`, `A`, and `AT` must be provided for DDNM sampling.")

        net = self.net
        t_steps = self.get_t_steps(latents).to(torch.float32)

        x_next = x_init.to(torch.float32) + latents.to(torch.float32) * t_steps[0]
        if self.save_path:
            save_nii_image(x_next, f"{self.save_path}/x_init.nii.gz")

        ATy = AT(y).to(torch.float32)

        x_t_list = []
        d_cur_list = []
        denoised_list = []

        pbar = tqdm.tqdm(range(len(t_steps) - 1), total=len(t_steps) - 1, colour="blue")
        for i in pbar:
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            x_cur = x_next

            denoised = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16).to(torch.float32)
            projected = AT(A(denoised)).to(torch.float32)
            x0_hat = ATy + denoised - projected

            d_cur = (x_cur - x0_hat) / t_cur  # score

            if self.save_intermediates:
                x_t_list.append(x_cur.detach().cpu().numpy())
                d_cur_list.append(d_cur.detach().cpu().numpy())
                denoised_list.append(x0_hat.detach().cpu().numpy())

            if float(t_next) > 0:
                noises = torch.randn_like(x0_hat)

                eta = 1

                sigma_sto = math.sqrt(eta) * t_next
                sigma_det = math.sqrt(1 - eta) * t_next

                noise_sto = sigma_sto * noises
                noise_det = sigma_det * d_cur

                x_next = x0_hat + noise_det + noise_sto
            else:
                x_next = x0_hat

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

        self.save_intermediate_results(x_t_list, d_cur_list, denoised_list)
        return x_next
