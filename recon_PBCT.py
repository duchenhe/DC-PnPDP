import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

import dnnlib
import utils
import utils.args
import utils.data
import utils.result
from algorithms import DCPnPDP, DiffPIR, base
from physics.ct import PBCT_carterbox


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
        f"{args.use_init}",
    ]
    if getattr(args, "num_cg", 0):
        parts.append(f"nCG-{args.num_cg}")
    if getattr(args, "w_dps", 0):
        parts.append(f"wDPS-{args.w_dps}")
    if getattr(args, "w_tik", 0):
        parts.append(f"wTIK-{args.w_tik}")
    if getattr(args, "w_dz", 0):
        parts.append(f"wDZ-{args.w_dz}")

    parts.append(str(args.noise_control))
    parts.append(str(args.renoise_method))

    return "_".join(map(str, parts))


def create_save_root(args, data_name, problem):
    save_root = Path(
        f"{args.save_dir}/{data_name}/PBCT/{args.task}-{args.degree}/{args.method}/{problem}/{datetime.datetime.now():%y%m%d_%H%M%S}/"
    )
    save_root.mkdir(parents=True, exist_ok=True)
    return save_root


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
    print(f'Loading network from "{ckpt_filename}"...')
    with dnnlib.util.open_url(ckpt_filename, verbose=True) as f:
        net = pickle.load(f)["ema"].to(device)
    return net


def run_reconstruction(args, net, save_root, measurement, fbp_lv, cg_lv, measure_model):
    latents = torch.randn_like(cg_lv.squeeze().unsqueeze(1))

    sampler_kwargs = {
        "net": net,
        "num_steps": args.NFE,
        "sigma_max": args.sigma_max,
        "save_path": save_root,
        "save_intermediates": True,
        "noise_control": args.noise_control,
    }
    recon_kwargs = {
        "latents": latents,
        "x_init": fbp_lv.clip(-1, 1).squeeze().unsqueeze(1),
        "y": measurement,
        "A": measure_model.A,
        "AT": measure_model.A_T,
        "num_cg": args.num_cg,
        "w_tik": args.w_tik,
    }

    if args.method == "edm":
        print("Using edm sampling.")
        sampler = base.BaseEDMSampler(**sampler_kwargs)
        x = sampler.sample(latents, x_init=cg_lv.squeeze().unsqueeze(1))
    elif args.method == "DiffPIR":
        print("Run DiffPIR!")
        sampler = DiffPIR.DiffPIR(**sampler_kwargs)
        x = sampler.sample(**recon_kwargs)
    elif args.method == "DCPnPDP":
        print("Run Dual Coupled DiffPIR!")
        sampler = DCPnPDP.DiffPIRDC(**sampler_kwargs)
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


def compute_and_save_metrics(save_root, fbp_lv, x, gt_image, metainfo, d, h, w):
    fbp_lv = reshape_for_metrics(fbp_lv, d, h, w)
    gt_image = reshape_for_metrics(gt_image, d, h, w)
    x = reshape_for_metrics(x, d, h, w)

    data_range_gt = (gt_image.max() - gt_image.min()).item()

    psnr, ssim = utils.result.cal_metrics(fbp_lv, gt_image, save_root / "FBP-LV_metrics")
    print("--------------------------------")
    print(f"FBP-LV PSNR: {psnr:.4f}\nFBP-LV SSIM: {ssim:.4f}")

    metrics = utils.result.compute_slice_metrics_optimized(fbp_lv, gt_image, data_range=data_range_gt)
    utils.result.print_slice_metrics(metrics)

    psnr, ssim = utils.result.cal_metrics(x, gt_image, save_root / "recon_metrics", sitk_info=metainfo)
    print("--------------------------------")
    print(f"recon PSNR: {psnr:.4f}\nrecon SSIM: {ssim:.4f}")

    import lpips

    lpips_net = lpips.LPIPS(net="squeeze").to("cuda")
    metrics = utils.result.compute_slice_metrics_optimized(
        x, gt_image, data_range=data_range_gt, lpips_batch_size=8, lpips_net=lpips_net
    )
    utils.result.print_slice_metrics(metrics, include_lpips=True)

    summary_metrics = {}
    target_keys = ["PSNR_mean", "SSIM_mean", "LPIPS_mean"]
    for axis in ["axial", "coronal", "sagittal"]:
        summary_metrics[axis] = {k: metrics[axis][k] for k in target_keys}

    with open(save_root / "recon_metrics" / "metrics_summary.yaml", "w") as f:
        yaml.dump(summary_metrics, f, sort_keys=False)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_name, gt_image, metainfo, measure_model, projections, measurement = setup_measurement(args, device)
    d, h, w = gt_image.shape[1], gt_image.shape[2], gt_image.shape[3]

    print(gt_image.shape)
    print(f"Projections shape: {projections.shape}, Measurement shape: {measurement.shape}")

    fbp_lv, fbp_fv, cg_lv = compute_fbp_and_cg(measure_model, measurement, projections)

    problem = build_problem_tag(args)
    print(f"Task: {problem}")

    save_root = create_save_root(args, data_name, problem)
    print(f"Save to: {save_root}")
    save_run_args(args, save_root)
    save_basic_outputs(save_root, measurement, fbp_lv, fbp_fv, gt_image, cg_lv, metainfo)

    net = load_model(args.checkpoint_path, device)
    x = run_reconstruction(args, net, save_root, measurement, fbp_lv, cg_lv, measure_model)

    print("💡 x:", x.shape)
    utils.result.save_nii_image(x, os.path.join(save_root, "recon.nii.gz"), sitk_info=metainfo)
    print(f"Save to: {save_root}")

    compute_and_save_metrics(save_root, fbp_lv, x, gt_image, metainfo, d, h, w)


if __name__ == "__main__":
    args = utils.args.build_parser()
    main(args)
