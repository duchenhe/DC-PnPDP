import argparse


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
    parser.add_argument("--use-init", type=bool, default=False, help="Whether to use the initial image")
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

    args = parser.parse_args()
    return args
