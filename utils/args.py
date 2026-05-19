import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def build_parser():
    parser = argparse.ArgumentParser(description="DIS.")

    # base
    parser.add_argument("--task", type=str, default="LACT", help="reconstruction task. Default: LACT.")
    parser.add_argument("--method", type=str, default="DDS", help="Reconstruction method. Default: DDS.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use. Default: 0.")

    # data
    parser.add_argument(
        "--data",
        type=str,
        default="../data/AAPM/L506.nii.gz",
        help="Path to input data file.",
    )
    parser.add_argument("--slice-begin", type=int, default=0, help="Slice index for reconstruction. Default: 0.")
    parser.add_argument("--slice-end", type=int, default=0, help="Slice index for reconstruction. Default: 0.")
    parser.add_argument("--slice-step", type=int, default=1, help="Slice step for reconstruction. Default: 1.")

    parser.add_argument("--sino-noise", type=float, default=0, help="Sinogram noise level. Default: 0.")
    parser.add_argument("--degree", type=int, default=90, help="Available projection degree. Default: 90.")
    parser.add_argument("--recon-size", type=int, default=256, help="Reconstruction image size. Default: 256.")
    parser.add_argument("--use-init", type=str2bool, default=False, help="Whether to use the initial image")
    parser.add_argument("--save_dir", type=str, default="results", help="Path to save results.")

    # algorithm
    parser.add_argument("--NFE", type=int, default=1000, help="Run steps for the algorithm. Default: 1000.")
    parser.add_argument("--num-cg", type=int, default=5, help="Number of CG iterations. Default: 5.")
    parser.add_argument("--w-dps", type=float, default=0, help="DPS regularization weight. Default: 0.025.")
    parser.add_argument("--w-tik", type=float, default=0, help="Tikhonov regularization weight. Default: 1.")
    parser.add_argument("--w-dz", type=float, default=0, help="TV regularization on Z axis weight. Default: 1.")
    parser.add_argument("--sigma-max", type=float, default=378, help="The maximum sigma value")
    parser.add_argument("--sigma-min", type=float, default=0.01, help="The minimum sigma value")
    parser.add_argument(
        "--sitcom-learning-rate",
        type=float,
        default=0.01,
        help="SITCOM inner Adam learning rate for optimizing the noisy sample.",
    )
    parser.add_argument(
        "--sitcom-dc-weight",
        type=float,
        default=1.0,
        help="SITCOM data-consistency loss weight.",
    )
    parser.add_argument(
        "--sitcom-denoise-batch-size",
        type=int,
        default=1,
        help="Batch size for SITCOM denoiser evaluation with autograd enabled.",
    )
    parser.add_argument(
        "--sitcom-clamp-denoised",
        type=str2bool,
        default=True,
        help="Whether to clamp SITCOM denoised predictions into [-1, 1].",
    )
    parser.add_argument("--daps-diffusion-num-steps", type=int, default=5, help="DAPS PF-ODE predictor steps.")
    parser.add_argument(
        "--daps-diffusion-sigma-min",
        type=float,
        default=0.01,
        help="DAPS minimum sigma used inside the local PF-ODE predictor.",
    )
    parser.add_argument("--daps-lgvd-num-steps", type=int, default=20, help="DAPS Langevin SGD inner steps.")
    parser.add_argument("--daps-lgvd-lr", type=float, default=5e-6, help="DAPS Langevin SGD learning rate.")
    parser.add_argument("--daps-lgvd-tau", type=float, default=0.25, help="DAPS data-fidelity noise scale tau.")
    parser.add_argument(
        "--daps-lgvd-lr-min-ratio",
        type=float,
        default=0.01,
        help="DAPS minimum LR ratio for annealed Langevin dynamics.",
    )
    parser.add_argument(
        "--daps-denoise-batch-size",
        type=int,
        default=16,
        help="Batch size for DAPS ODE denoiser evaluation.",
    )
    parser.add_argument("--red-lr", type=float, default=0.5, help="RED-Diff Adam learning rate.")
    parser.add_argument("--red-lambda", type=float, default=None, help="RED-Diff denoiser regularization weight.")
    parser.add_argument(
        "--red-obs-weight",
        type=float,
        default=None,
        help="RED-Diff observation consistency weight. Falls back to --w-dps or 1.0.",
    )
    parser.add_argument(
        "--red-lambda-schedule",
        type=str,
        default="constant",
        help="RED-Diff denoiser-weight schedule: constant, linear, sqrt, square, log, trunc_linear, power2over3.",
    )
    parser.add_argument(
        "--red-denoise-batch-size",
        type=int,
        default=16,
        help="Batch size for RED-Diff denoiser evaluation.",
    )

    parser.add_argument(
        "--noise-control", type=str, default=None, help="Type of noise to add to sinogram. Default: gaussian."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="../checkpoint/checkpoint.pth",
        help="The path to the checkpoint",
    )
    parser.add_argument("--config-path", type=str, default="configs/config.yaml", help="The path to the config file")
    parser.add_argument("--renoise-method", type=str, default="DDPM", help="The re-noising method")
    parser.add_argument(
        "--metric-axes",
        type=str,
        default="axial",
        choices=["axial", "all"],
        help="Which axes to use when computing slice metrics: axial only, or all three axes.",
    )
    parser.add_argument(
        "--save-residual-history",
        type=str2bool,
        default=True,
        help="Whether to save per-iteration convergence residuals for methods that support it.",
    )
    parser.add_argument(
        "--save-runtime-profile",
        type=str2bool,
        default=True,
        help="Whether to save per-iteration wall-clock and memory profiling for methods that support it.",
    )

    args = parser.parse_args()
    return args
