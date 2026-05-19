from collections.abc import Callable

import torch
import tqdm

from algorithms.base import BaseEDMSampler
import algorithms.utils as autils
from utils.result import save_nii_image

LinearOp = Callable[[torch.Tensor], torch.Tensor]


class SITCOM(BaseEDMSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _denoise_with_grad_batchwise(self, net, x, t_cur, class_labels, batch_size=None):
        """Batch the denoiser forward pass while keeping autograd enabled."""
        if batch_size is None:
            batch_size = max(1, min(self._estimate_batch_size(x) // 2, x.shape[0], 8))

        batches = autils.batchfy(x, batch_size)

        if torch.is_tensor(class_labels) and class_labels.ndim > 0 and class_labels.shape[0] == x.shape[0]:
            label_batches = autils.batchfy(class_labels, batch_size)
        else:
            label_batches = [class_labels] * len(batches)

        denoised_batches = [net(xb, t_cur, lb).to(torch.float32) for xb, lb in zip(batches, label_batches)]
        return torch.cat(denoised_batches, dim=0)

    def _optimize_noisy_sample(
        self,
        x_start,
        t_cur,
        net,
        y,
        A,
        class_labels,
        num_steps,
        learning_rate,
        denoise_batch_size=1,
        dc_weight=1.0,
        w_tik=0.0,
        clamp_denoised=False,
    ):
        x_opt = x_start.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_opt], lr=learning_rate)
        anchor = x_start.detach()
        loss_value = float("nan")
        denoise_time_total = 0.0
        device = x_start.device
        dc_start = self._timer_start(device)

        for _ in range(num_steps):
            optimizer.zero_grad(set_to_none=True)

            denoise_start = self._timer_start(device)
            x0_pred = self._denoise_with_grad_batchwise(
                net,
                x_opt,
                t_cur,
                class_labels,
                batch_size=denoise_batch_size,
            )
            denoise_time_total += self._timer_end(denoise_start, device)
            if clamp_denoised:
                x0_pred = x0_pred.clamp(-1, 1)

            residual = A(x0_pred) - y
            loss = dc_weight * torch.linalg.vector_norm(residual)

            if w_tik > 0:
                loss = loss + w_tik * torch.linalg.vector_norm(x_opt - anchor)

            loss.backward()
            optimizer.step()
            loss_value = loss.detach().item()

        with torch.no_grad():
            denoise_start = self._timer_start(device)
            x0_pred = self._denoise_batchwise(net, x_opt, t_cur, class_labels, batch_size=denoise_batch_size)
            denoise_time_total += self._timer_end(denoise_start, device)
            if clamp_denoised:
                x0_pred = x0_pred.clamp(-1, 1)

        dc_time = self._timer_end(dc_start, device)
        return x_opt.detach(), x0_pred.detach(), loss_value, {
            "dc_time_s": dc_time,
            "denoise_time_s": denoise_time_total,
        }

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
        learning_rate=0.01,
        dc_weight=1.0,
        denoise_batch_size=None,
        clamp_denoised=True,
    ):
        if y is None or A is None or AT is None:
            raise ValueError("`y`, `A`, and `AT` must be provided for SITCOM sampling.")

        # Reuse `num_cg` as the number of inner optimization steps to keep the
        # existing reconstruction entrypoint compatible with the SITCOM interface.
        inner_steps = max(int(num_cg), 1)

        net = self.net
        t_steps = self.get_t_steps(latents).to(torch.float32)

        x_next = x_init.to(torch.float32) + latents.to(torch.float32) * t_steps[0]
        if self.save_path:
            save_nii_image(x_next, f"{self.save_path}/x_init.nii.gz")

        x_t_list = []
        d_cur_list = []
        denoised_list = []
        runtime_history = {
            "method": "SITCOM",
            "definitions": {
                "outer_iter_time_s": "Wall-clock time of one outer iteration.",
                "denoise_time_s": "Accumulated denoiser forward time inside the SITCOM inner optimization and final denoise.",
                "dc_time_s": "Total time spent in the SITCOM inner optimization sub-problem.",
                "sh_time_s": "Spectral homogenization time. Always zero for SITCOM.",
                "other_time_s": "Outer iteration time minus the SITCOM inner optimization time.",
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

            _, x0_cur, loss_value, timing = self._optimize_noisy_sample(
                x_start=x_cur,
                t_cur=t_cur,
                net=net,
                y=y,
                A=A,
                class_labels=class_labels,
                num_steps=inner_steps,
                learning_rate=learning_rate,
                denoise_batch_size=denoise_batch_size,
                dc_weight=dc_weight,
                w_tik=w_tik,
                clamp_denoised=clamp_denoised,
            )

            if self.save_intermediates:
                x_t_list.append(x_cur.detach().cpu().numpy())
                d_cur_list.append((x_cur - x0_cur).detach().cpu().numpy())
                denoised_list.append(x0_cur.detach().cpu().numpy())

            if float(t_next) > 0:
                if str(self.noise_control).lower() in {"deterministic", "zero"}:
                    noise = torch.zeros_like(x0_cur)
                else:
                    noise = torch.randn_like(x0_cur)
                x_next = x0_cur + t_next * noise
            else:
                x_next = x0_cur

            outer_time = self._timer_end(outer_start, x_cur.device)
            runtime_history["records"].append(
                {
                    "iteration": i + 1,
                    "sigma": float(t_cur),
                    "outer_iter_time_s": outer_time,
                    "denoise_time_s": timing["denoise_time_s"],
                    "dc_time_s": timing["dc_time_s"],
                    "sh_time_s": 0.0,
                    "other_time_s": max(outer_time - timing["dc_time_s"], 0.0),
                    "peak_memory_mb": self._get_peak_memory_mb(x_cur.device),
                }
            )

            pbar.set_postfix({"t": i, "σ_t": f"{float(t_cur):.4f}", "dc": f"{loss_value:.3e}"})

        self.save_intermediate_results(x_t_list, d_cur_list, denoised_list)
        self._save_runtime_profile(runtime_history)
        return x_next
