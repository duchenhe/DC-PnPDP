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

        for i in pbar:  # 0, ..., N-1
            t_cur = t_steps[i]

            rho_tik = w_tik * (1 / t_cur**2).item()

            # * 1. x update， data sub-problem
            x_in = v - u

            bcg = b_cg(x_in, rho_tik)

            x = autils.cg_uni(Acg_fn, bcg, x_in, rho=rho_tik, maxiter=num_cg)
            # *----------------------------------------------------------------------------

            # * 2. v update， prior sub-problem
            x_cur = x + u

            # if i > 0:
            if i != len(t_steps) - 1:
                x_cur, info = spectral_homogenization_2d_batched(
                    v=x_cur,
                    x_hat=v,
                    sigma=t_cur,
                    batch_size=50,
                    smooth_ks=7,
                    eps_ratio=1e-6,
                    aggregate="batch_mean",
                )

            v = self._denoise_batchwise(net, x_cur, t_cur, class_labels, batch_size=16)

            # *----------------------------------------------------------------------------

            # * 3. u update
            u = u + (x - v)
            # *----------------------------------------------------------------------------

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}"})

        return v
