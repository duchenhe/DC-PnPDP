import dnnlib
import pickle
import torch
import numpy as np
import SimpleITK as sitk
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from algorithms import DiffPIR, PnP_ADMM_Diff, DDS_eq, ADMM_DM, ADMM_DM_awgn, DDNM, DDS, DAPS
from utils.result import save_nii_image, cal_metrics
from utils.data import nchw_comp_to_real, real_to_nchw_comp
from physics.mri import SinglecoilMRI_comp
import jsmoco_utils
from algo import diffpir

torch.set_num_threads(20)

# 参数解析
parser = argparse.ArgumentParser(description="DIS.")
parser.add_argument("--method", type=str, default="DDS", help="Method to use for reconstruction.")
parser.add_argument("--NFE", type=int, default=100, help="Number of function evaluations.")
parser.add_argument("--sigma_max", type=float, default=10.0, help="Maximum noise level for EDM.")
parser.add_argument("--save_dir", type=str, default="./results/singleCoil", help="Directory to save results.")
parser.add_argument("--num_cg", type=int, default=5, help="Number of CG iterations.")
parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
parser.add_argument("--task", type=str, default="knee", help="Task: brain or knee.")
parser.add_argument("--w_tik", type=float, default=0.0, help="Weight for Tikhonov regularization.")
parser.add_argument("--ACS", type=int, default=24, help="ACS size for undersampling.")
parser.add_argument("--AF", type=int, default=4, help="Acceleration factor.")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# 加载网络
ckpt_filename = f"/mnt/MIXVol-1/dch/checkpoint/edm/InverseBench/MRI-{args.task}.pt"
print(f'Loading network from "{ckpt_filename}"...')

config = OmegaConf.load(f"/mnt/sda/dch/code/DPER/InverseBench/configs/pretrain/mri-{args.task}-mvue.yaml")
network_kwargs = dnnlib.EasyDict(config.model)
network_kwargs.pop("_target_", None)
network_kwargs.class_name = "training.networks.EDMPrecond"

try:
    with dnnlib.util.open_url(ckpt_filename, "rb") as f:
        net = pickle.load(f)["ema"].to(device)
except:
    net = dnnlib.util.construct_class_by_name(**network_kwargs)
    ckpt = torch.load(ckpt_filename, map_location=device, weights_only=True)
    net.load_state_dict(ckpt.get("ema", ckpt["net"]))
    net = net.to(device)

print("Successfully loaded the model.")

# 设置保存路径
save_path = Path(f"./{args.save_dir}/mri_{args.task}/{args.method}/AF_{args.AF}_ACS_{args.ACS}")

# 加载数据
mvue = sitk.ReadImage(f"./data/inversebench_{args.task}/mvue_aggregate.nii.gz")
sitk_info = {"spacing": mvue.GetSpacing(), "direction": mvue.GetDirection()}
mvue = sitk.GetArrayFromImage(mvue)
nBc, nRd, nPe = mvue.shape

gt_img = torch.from_numpy(mvue).view(nBc, 1, nRd, nPe).to(device)
print(f"mvue shape: {gt_img.shape}, dtype: {gt_img.dtype}")

# # 只取batch里的后一半
# gt_img = gt_img[nBc // 2 :, ...]
# nBc = gt_img.shape[0]
# gt_img = gt_img[::20, ...]
# sitk_info = None

# nBc = gt_img.shape[0]

# 创建采样掩码
mask = jsmoco_utils.KspaceUnd(nRd, nPe, Rx=1, Ry=args.AF, ACSx=nRd, ACSy=args.ACS)
mask = torch.from_numpy(mask[None, None, ...]).to(torch.int16).to(device)
print(f"Mask shape: {mask.shape}")

# 测量模型
measure_model = SinglecoilMRI_comp(mask=mask)

# 前向和伴随操作
y = measure_model.A(gt_img)
ATy = measure_model.AT(y)
ATy = ATy / torch.abs(ATy).max()

# 保存中间结果
save_nii_image(y, f"{save_path}/y.nii.gz", sitk_info=sitk_info)
save_nii_image(ATy, f"{save_path}/ATy.nii.gz", sitk_info=sitk_info)
save_nii_image(gt_img, f"{save_path}/gt.nii.gz", sitk_info=sitk_info)
save_nii_image(mask, f"{save_path}/mask.nii.gz")

# 采样重建
sampler_kwargs = dict(
    num_steps=args.NFE, sigma_max=args.sigma_max, save_path=save_path, noise_control="None", save_intermediates=False
)
sample_kwargs = dict(
    x_init=nchw_comp_to_real(ATy).view(nBc, 2, nRd, nPe).to(device),
    y=y,
    A=measure_model.A,
    AT=measure_model.AT,
    num_cg=args.num_cg,
    w_tik=args.w_tik,
)

with torch.no_grad():
    if args.method == "DiffPIR":
        print("Run DiffPIR!")
        sampler = DiffPIR.DiffPIRSampler(net, save_slices=10, **sampler_kwargs)
    elif args.method == "ADMM-DM":
        print("Run PnP-ADMM-DM!")
        sampler = ADMM_DM.DiffPnPADMMSampler(net, save_slices=10, **sampler_kwargs)
    elif args.method == "ADMM-DM-AWGN":
        print("Run PnP-ADMM-DM-AWGN!")
        sampler = ADMM_DM_awgn.DiffPnPADMMSampler(net, save_slices=10, **sampler_kwargs)

    elif args.method == "DDNM":  # OK
        print("Run DDNM!")
        sampler = DDNM.DDNMSampler(net, **sampler_kwargs)

    elif args.method == "DDS":  # OK
        print("Run DDS!")
        sampler = DDS.DiffPIRSampler(net, **sampler_kwargs)

    elif args.method == "DAPS":  # OK
        print("Run DAPS!")
        sampler = DAPS.DAPS(net, **sampler_kwargs)

    elif args.method == "algo-DiffPIR":
        print("Run algo-DiffPIR!")
        method_config = OmegaConf.load("configs/diffpir.yaml").method
        print(f"Method config: {method_config}")
        # 使用 ** 解包配置参数
        sampler = diffpir.DiffPIR(net=net, **method_config)
    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")
    x = sampler.sample(**sample_kwargs)

# 后处理和保存结果
x = real_to_nchw_comp(x)
save_nii_image(x, f"{save_path}/recon.nii.gz", sitk_info=sitk_info)

x, gt, ATy = torch.abs(x), torch.abs(gt_img), torch.abs(ATy)


# 最大最小归一化
def normalize(t):
    return (t - t.min()) / (t.max() - t.min())


def normalize_percentile(img, percentile=99):
    """
    使用百分位数进行归一化，支持 numpy 数组和 torch 张量

    Args:
        img: numpy 数组或 torch 张量
        percentile: 百分位数 (0-100)

    Returns:
        归一化后的图像，类型与输入相同
    """
    if isinstance(img, torch.Tensor):
        scaling = torch.quantile(torch.abs(img.flatten()), percentile / 100.0)
        return img / scaling
    else:
        scaling = np.quantile(np.abs(img), percentile / 100.0)
        return img / scaling


# x, gt, ATy = normalize(x), normalize(gt), normalize(ATy)
x, gt, ATy = normalize_percentile(x), normalize_percentile(gt), normalize_percentile(ATy)

# 保存最终结果
suffix = f"_{args.method}_AF_{args.AF}_ACS_{args.ACS}.nii.gz"
save_nii_image(gt, save_path / f"gt_all{suffix}", sitk_info=sitk_info)
save_nii_image(x, save_path / f"recon_all{suffix}", sitk_info=sitk_info)
save_nii_image(ATy, save_path / f"ATy_all{suffix}", sitk_info=sitk_info)

# 计算指标
print(f"x shape: {x.shape}, gt shape: {gt.shape}")
metrics = cal_metrics(x.to(device), gt.to(device), save_path=save_path / "metrics/", sitk_info=sitk_info)
print(f"PSNR: {metrics[0]:.4f}, SSIM: {metrics[1]:.4f}, MS-SSIM: {metrics[2]:.4f}")
