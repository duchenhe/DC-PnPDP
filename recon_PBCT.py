import datetime
import io
import os
import pickle
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import yaml

import utils
import utils.args
import utils.data
import utils.result
from algorithms import DAPS, DCPnPDP, DDNM, DDS, DiffPIR, SITCOM, base
from physics.ct import PBCT_carterbox

from rich import print

# Runtime settings
torch.set_num_threads(20)
HU_MAX = 800
HU_MIN = -800


def get_view_indices(task, degree, view_full_num):
    if task == "LACT":
        view_limited_num = int(view_full_num * (degree / 180))
        view_limited_idx = np.linspace(0, view_limited_num, view_limited_num, endpoint=False, dtype=int)
    elif task == "SVCT":
        view_limited_num = degree
        view_limited_idx = np.linspace(0, view_full_num, view_limited_num, endpoint=False, dtype=int)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return view_limited_idx


def add_noise_if_needed(measurement, sino_noise):
    if sino_noise > 0:
        level = np.sqrt(sino_noise) if sino_noise < 100 else sino_noise
        noise_type = "gaussian" if sino_noise < 100 else "poisson"
        measurement, snr = utils.data.add_sino_noise_guassian(measurement, level, noise_type)
        print(f"Add {noise_type} Noise to Measurement with level: {level} and SNR: {snr}")
    return measurement


def build_problem_tag(args):
    parts = [
        f"{args.task}",
        f"{args.degree}",
        f"{args.slice_begin:04d}",
        f"{args.slice_end:04d}",
        f"{args.slice_step:03d}",
        f"{args.NFE}",
    ]
    parts.append(f"sigMin-{args.sigma_min:g}")
    parts.append(f"sigMax-{args.sigma_max:g}")
    if getattr(args, "num_cg", 0):
        parts.append(f"nCG-{args.num_cg}")
    if getattr(args, "w_tik", 0):
        parts.append(f"wTIK-{args.w_tik}")
    parts.append(f"metricAxes-{args.metric_axes}")
    if args.method == "DAPS":
        parts.append(f"dapsODE-{args.daps_diffusion_num_steps}")
        parts.append(f"dapsODESigMin-{args.daps_diffusion_sigma_min}")
        parts.append(f"dapsLGSteps-{args.daps_lgvd_num_steps}")
        parts.append(f"dapsLR-{args.daps_lgvd_lr}")
        parts.append(f"dapsTau-{args.daps_lgvd_tau}")
    if args.method == "SITCOM":
        parts.append(f"sitLR-{args.sitcom_learning_rate}")
        parts.append(f"sitDC-{args.sitcom_dc_weight}")
        parts.append(f"sitBS-{args.sitcom_denoise_batch_size}")
        parts.append(f"sitClamp-{args.sitcom_clamp_denoised}")

    parts.append(str(args.noise_control))

    return "_".join(map(str, parts))


def create_save_root(args, data_name, problem):
    save_root = Path(
        f"{args.save_dir}/{data_name}/PBCT/{args.task}-{args.degree}/{args.method}/{problem}/{datetime.datetime.now():%y%m%d_%H%M%S_%f}/"
    )
    save_root.mkdir(parents=True, exist_ok=True)
    return save_root


def print_run_header(args, save_root: Path, problem_tag: str):
    data_path = Path(args.data).resolve()
    print("[bold cyan]🚀 Starting PBCT Reconstruction[/bold cyan]")
    print(f"🧠 Data: [bold]{data_path}[/bold]")
    print(f"🧪 Method: [bold]{args.method}[/bold] | Task: [bold]{args.task}[/bold] | GPU: [bold]{args.gpu}[/bold]")
    print(
        f"📚 Slices: [bold]{args.slice_begin}:{args.slice_end}:{args.slice_step}[/bold] | "
        f"Recon size: [bold]{args.recon_size}[/bold]"
    )
    print(f"🌀 CT views: degree=[bold]{args.degree}[/bold] | sino_noise=[bold]{args.sino_noise:g}[/bold]")
    print(
        f"⚙️  EDM: NFE=[bold]{args.NFE}[/bold], num_cg=[bold]{args.num_cg}[/bold], "
        f"sigma=[bold]{args.sigma_min:g} -> {args.sigma_max:g}[/bold]"
    )
    if args.method == "DAPS":
        print(
            f"🧮 DAPS: pf_steps=[bold]{args.daps_diffusion_num_steps}[/bold], "
            f"pf_sigma_min=[bold]{args.daps_diffusion_sigma_min:g}[/bold], "
            f"lg_steps=[bold]{args.daps_lgvd_num_steps}[/bold], "
            f"lg_lr=[bold]{args.daps_lgvd_lr:g}[/bold], "
            f"tau=[bold]{args.daps_lgvd_tau:g}[/bold], "
            f"lr_ratio=[bold]{args.daps_lgvd_lr_min_ratio:g}[/bold]"
        )
    elif args.method == "DDS":
        print("🧮 DDS: uses EDM denoiser + CG-based data consistency with stochastic re-noising.")
    elif args.method == "DDNM":
        print("🧮 DDNM-SIRT: uses SIRT pseudo-inverse correction with stochastic re-noising.")
    elif args.method == "SITCOM":
        print(
            f"🧮 SITCOM: lr=[bold]{args.sitcom_learning_rate:g}[/bold], "
            f"dc_weight=[bold]{args.sitcom_dc_weight:g}[/bold], "
            f"batch=[bold]{args.sitcom_denoise_batch_size}[/bold], "
            f"clamp=[bold]{args.sitcom_clamp_denoised}[/bold]"
        )
    print(f"🏷️  Problem: [bold]{problem_tag}[/bold]")
    print(f"💾 Save root: [bold]{save_root}[/bold]")
    print("")


def save_run_args(args, save_root):
    with open(save_root / "args.yaml", "w") as f:
        yaml.dump(vars(args), f)


def setup_measurement(args, device):
    data_path = Path(args.data)
    data_name = data_path.stem

    view_full_num = 360
    view_limited_idx = get_view_indices(args.task, args.degree, view_full_num)

    gt_image, metainfo = utils.data.load_and_preprocess_image(args, HU_MAX, HU_MIN, device=device)

    measure_model = PBCT_carterbox(
        det_count=363,
        view_available=view_limited_idx,
        view_full_num=view_full_num,
        recon_size=args.recon_size,
    )

    projections = measure_model.A_FV(gt_image).float().to(device).detach()
    measurement = measure_model.A(gt_image).float().to(device).detach()
    measurement = add_noise_if_needed(measurement, args.sino_noise)

    return data_name, gt_image, metainfo, measure_model, projections, measurement


def compute_fbp_and_cg(measure_model, measurement, projections):
    fbp_lv = measure_model.A_dagger(measurement)
    fbp_fv = measure_model.FBP_FV(projections)

    bcg = measure_model.A_T(measurement)
    import algorithms.utils as autils

    def A_cg(x, rho_tik):
        return measure_model.A_T(measure_model.A(x))

    cg_lv = autils.cg_uni(A_fn=A_cg, b=bcg, rho=0, maxiter=200)

    return fbp_lv, fbp_fv, cg_lv


def load_model(ckpt_filename, device):
    print(f'🧠 Loading network from [bold]"{ckpt_filename}"[/bold]...')
    with open(ckpt_filename, "rb") as f:
        return pickle.load(f)["ema"].to(device)


def load_lpips_quietly(device):
    import lpips

    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13.*")
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13.*",
        )
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        with redirect_stdout(buffer), redirect_stderr(buffer):
            return lpips.LPIPS(net="squeeze").to(device)


def run_reconstruction(args, net, save_root, measurement, fbp_lv, cg_lv, measure_model):
    # `cg_lv` is already in [B, 1, H, W]. Avoid `squeeze()` here because
    # single-slice chunks would collapse the batch dimension and break the U-Net
    # input layout.
    latents = torch.randn_like(cg_lv)

    sampler_kwargs = {
        "net": net,
        "num_steps": args.NFE,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "save_path": save_root,
        "save_intermediates": False,
        "noise_control": args.noise_control,
    }
    recon_kwargs = {
        "latents": latents,
        "x_init": fbp_lv.clip(-1, 1),
        "y": measurement,
        "A": measure_model.A,
        "AT": measure_model.A_T,
        "num_cg": args.num_cg,
        "w_tik": args.w_tik,
    }

    if args.method == "edm":
        print("🎲 Using EDM sampling.")
        sampler = base.BaseEDMSampler(**sampler_kwargs)
        x = sampler.sample(latents, x_init=cg_lv)
    elif args.method == "DiffPIR":
        print("🧪 Running DiffPIR...")
        sampler = DiffPIR.DiffPIR(**sampler_kwargs)
        x = sampler.sample(**recon_kwargs)
    elif args.method == "DDNM":
        print("🧪 Running DDNM-SIRT...")
        sampler = DDNM.DDNM(**sampler_kwargs)
        x = sampler.sample(**recon_kwargs)
    elif args.method == "DDS":
        print("🧪 Running DDS...")
        sampler = DDS.DDS(**sampler_kwargs)
        x = sampler.sample(**recon_kwargs)
    elif args.method == "DCPnPDP":
        print("🧪 Running DC-PnPDP...")
        sampler = DCPnPDP.DCPnPDP(**sampler_kwargs)
        x = sampler.sample(**recon_kwargs)
    elif args.method == "SITCOM":
        print("🧪 Running SITCOM...")
        sampler = SITCOM.SITCOM(**sampler_kwargs)
        x = sampler.sample(
            **recon_kwargs,
            learning_rate=args.sitcom_learning_rate,
            dc_weight=args.sitcom_dc_weight,
            denoise_batch_size=args.sitcom_denoise_batch_size,
            clamp_denoised=args.sitcom_clamp_denoised,
        )
    elif args.method == "DAPS":
        print("🧪 Running DAPS...")
        sampler = DAPS.DAPS(
            **sampler_kwargs,
            diffusion_num_steps=args.daps_diffusion_num_steps,
            diffusion_sigma_min=args.daps_diffusion_sigma_min,
            lgvd_num_steps=args.daps_lgvd_num_steps,
            lgvd_lr=args.daps_lgvd_lr,
            lgvd_tau=args.daps_lgvd_tau,
            lgvd_lr_min_ratio=args.daps_lgvd_lr_min_ratio,
            denoise_batch_size=args.daps_denoise_batch_size,
        )
        x = sampler.sample(**recon_kwargs)
    else:
        raise ValueError(f"Invalid method: {args.method}.")

    return x


def save_basic_outputs(save_root, measurement, fbp_lv, fbp_fv, gt_image, cg_lv, metainfo):
    utils.result.save_nii_image(measurement, os.path.join(save_root, "measurement.nii.gz"))
    utils.result.save_nii_image(fbp_lv, os.path.join(save_root, "FBP-LV.nii.gz"), sitk_info=metainfo)
    utils.result.save_nii_image(fbp_fv, os.path.join(save_root, "FBP-FV.nii.gz"), sitk_info=metainfo)
    utils.result.save_nii_image(gt_image, os.path.join(save_root, "GT.nii.gz"), sitk_info=metainfo)
    utils.result.save_nii_image(cg_lv, os.path.join(save_root, "CG-LV.nii.gz"), sitk_info=metainfo)


def reshape_for_metrics(x, d, h, w):
    return x.view(d, 1, h, w).clip(-1, 1)


def resolve_metric_axes(metric_axes: str):
    if metric_axes == "axial":
        return ("axial",)
    if metric_axes == "all":
        return ("axial", "coronal", "sagittal")
    raise ValueError(f"Unsupported metric_axes: {metric_axes}")


def compute_and_save_metrics(save_root, fbp_lv, x, gt_image, metainfo, d, h, w, metric_axes="all"):
    fbp_lv = reshape_for_metrics(fbp_lv, d, h, w)
    gt_image = reshape_for_metrics(gt_image, d, h, w)
    x = reshape_for_metrics(x, d, h, w)
    selected_axes = resolve_metric_axes(metric_axes)

    data_range_gt = (gt_image.max() - gt_image.min()).item()

    print("")
    print("[bold cyan]📊 Evaluating Reconstruction[/bold cyan]")
    print(f"🗂️  Metrics dir: [bold]{save_root / 'recon_metrics'}[/bold]")
    print(f"🧭 Metric axes: [bold]{', '.join(selected_axes)}[/bold]")

    psnr, ssim = utils.result.cal_metrics(fbp_lv, gt_image, save_root / "FBP-LV_metrics")
    print("")
    print("[bold]📌 FBP-LV Baseline[/bold]")
    print(f"PSNR: [bold]{psnr:.4f}[/bold]")
    print(f"SSIM: [bold]{ssim:.4f}[/bold]")

    metrics = utils.result.compute_slice_metrics_optimized(
        fbp_lv, gt_image, data_range=data_range_gt, axes=selected_axes
    )
    utils.result.print_slice_metrics(metrics, axes=selected_axes)

    psnr, ssim = utils.result.cal_metrics(x, gt_image, save_root / "recon_metrics", sitk_info=metainfo)
    print("")
    print("[bold]✨ Reconstruction[/bold]")
    print(f"PSNR: [bold green]{psnr:.4f}[/bold green]")
    print(f"SSIM: [bold green]{ssim:.4f}[/bold green]")

    lpips_net = load_lpips_quietly(x.device)
    metrics = utils.result.compute_slice_metrics_optimized(
        x,
        gt_image,
        data_range=data_range_gt,
        lpips_batch_size=8,
        lpips_net=lpips_net,
        axes=selected_axes,
    )
    utils.result.print_slice_metrics(metrics, include_lpips=True, axes=selected_axes)

    summary_metrics = {}
    target_keys = ["PSNR_mean", "SSIM_mean", "LPIPS_mean"]
    for axis in selected_axes:
        summary_metrics[axis] = {k: metrics[axis][k] for k in target_keys}

    with open(save_root / "recon_metrics" / "metrics_summary.yaml", "w") as f:
        yaml.dump(summary_metrics, f, sort_keys=False)
    print(f"📝 Saved metric summary to [bold]{save_root / 'recon_metrics' / 'metrics_summary.yaml'}[/bold]")


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_name, gt_image, metainfo, measure_model, projections, measurement = setup_measurement(args, device)
    d, h, w = gt_image.shape[1], gt_image.shape[2], gt_image.shape[3]

    problem = build_problem_tag(args)
    save_root = create_save_root(args, data_name, problem)
    print_run_header(args, save_root, problem)

    print("[bold cyan]📦 Measurement Setup[/bold cyan]")
    print(f"🖼️  GT shape: [bold]{tuple(gt_image.shape)}[/bold]")
    print(f"📡 Projections shape: [bold]{tuple(projections.shape)}[/bold]")
    print(f"📏 Measurement shape: [bold]{tuple(measurement.shape)}[/bold]")

    print("")
    print("[bold cyan]🧰 Classical Reconstructions[/bold cyan]")
    fbp_lv, fbp_fv, cg_lv = compute_fbp_and_cg(measure_model, measurement, projections)
    print(f"FBP-LV: [bold]{tuple(fbp_lv.shape)}[/bold] | FBP-FV: [bold]{tuple(fbp_fv.shape)}[/bold] | CG-LV: [bold]{tuple(cg_lv.shape)}[/bold]")

    save_run_args(args, save_root)
    save_basic_outputs(save_root, measurement, fbp_lv, fbp_fv, gt_image, cg_lv, metainfo)
    print(f"💾 Saved inputs and baselines to [bold]{save_root}[/bold]")

    net = load_model(args.checkpoint_path, device)
    print("")
    print("[bold cyan]🧠 Diffusion Reconstruction[/bold cyan]")
    x = run_reconstruction(args, net, save_root, measurement, fbp_lv, cg_lv, measure_model)

    print(f"💡 Reconstruction tensor: [bold]{tuple(x.shape)}[/bold]")
    utils.result.save_nii_image(x, os.path.join(save_root, "recon.nii.gz"), sitk_info=metainfo)
    print(f"📝 Saved reconstruction to [bold]{save_root / 'recon.nii.gz'}[/bold]")

    if args.skip_metrics:
        print("[yellow]Skipping reconstruction metrics (--skip-metrics=True).[/yellow]")
    else:
        compute_and_save_metrics(save_root, fbp_lv, x, gt_image, metainfo, d, h, w, metric_axes=args.metric_axes)
    print("")
    print(f"[bold green]✅ Finished. Results saved under[/bold green] [bold]{save_root}[/bold]")


if __name__ == "__main__":
    args = utils.args.build_parser()
    main(args)
