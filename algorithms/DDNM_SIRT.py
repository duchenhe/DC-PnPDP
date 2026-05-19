from collections.abc import Callable

import math
import torch
import tqdm

from algorithms.SIRT import SIRTPseudoInverse
from algorithms.base import BaseEDMSampler
from utils.result import save_nii_image

LinearOp = Callable[[torch.Tensor], torch.Tensor]


class DDNM(BaseEDMSampler):
    def __init__(self, *args, eta: float = 1, **kwargs):
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
        num_cg=20,
        w_tik=0.0,
    ):
        del w_tik  # kept for a shared entrypoint signature

        if y is None or A is None or AT is None:
            raise ValueError("`y`, `A`, and `AT` must be provided for DDNM-SIRT sampling.")

        net = self.net
        t_steps = self.get_t_steps(latents).to(torch.float32)

        x_next = x_init.to(torch.float32) + latents.to(torch.float32) * t_steps[0]
        if self.save_path:
            save_nii_image(x_next, f"{self.save_path}/x_init.nii.gz")

        sirt = SIRTPseudoInverse(A, AT, vol_shape=x_next.shape, device=x_next.device, dtype=x_next.dtype)
        x_data = sirt.pinv(y, x0=None, n_iter=num_cg, lam=1.0, min_value=-1).to(torch.float32)

        x_t_list = []
        d_cur_list = []
        denoised_list = []

        pbar = tqdm.tqdm(range(len(t_steps) - 1), total=len(t_steps) - 1, colour="blue")
        for i in pbar:
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            x_cur = x_next

            denoised = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16).to(torch.float32)

            y_hat = A(denoised)
            x_proj = sirt.pinv(y_hat, x0=None, n_iter=num_cg, lam=1.0, min_value=-1).to(torch.float32)
            x_0_cur_hat = denoised - x_proj + x_data

            d_cur = (x_cur - x_0_cur_hat) / t_cur

            if self.save_intermediates:
                x_t_list.append(x_cur.detach().cpu().numpy())
                d_cur_list.append(d_cur.detach().cpu().numpy())
                denoised_list.append(x_0_cur_hat.detach().cpu().numpy())

            if float(t_next) > 0:
                noise = torch.randn_like(x_0_cur_hat)
                sigma_sto = self.eta * t_next
                sigma_det = math.sqrt(max(1.0 - self.eta**2, 0.0)) * t_next
                x_next = x_0_cur_hat + sigma_det * d_cur + sigma_sto * noise
            else:
                x_next = x_0_cur_hat

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

        self.save_intermediate_results(x_t_list, d_cur_list, denoised_list)
        return x_next.clip(-1, 1)
