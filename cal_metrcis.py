import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 设置可见的GPU设备

import SimpleITK as sitk

import numpy as np
from pathlib import Path

# import DPER_utils
import torch
import yaml
import utils
import utils.args
import utils.data
import utils.result

from rich import print

data_dir = Path("./results/SVCT_20_DDS_256_Case_00066_0000_1_2000_0_300.0_0.001/250724_184334")
data_dir = Path(
    "./results/Case_00066_0000.nii/PBCT/SVCT-20/SITCOM/SVCT_20_0000_0500_100_50_True_nCG-50_None_DDPM/260325_193009"
)
data_dir = Path(
    "./results/Case_00066_0000.nii/PBCT/SVCT-20/DCPnPDP/SVCT_20_0000_0500_100_50_True_nCG-10_None_DDPM/260325_195458"
)
data_dir = Path(
    "./results/Case_00066_0000.nii/PBCT/SVCT-20/DCPnPDP/SVCT_20_0000_0500_100_50_True_nCG-50_None_DDPM/260325_195637"
)

data_dir = Path(
    "results/Case_00066_0000.nii/PBCT/SVCT-20/DiffPIR/SVCT_20_0000_0500_100_50_True_nCG-50_None_DDPM/260325_195743"
)
data_dir = Path(
    "results/Case_00066_0000.nii/PBCT/SVCT-20/DCPnPDP/SVCT_20_0000_0500_100_100_True_nCG-10_None_DDPM/260325_201150"
)
data_dir = Path(
    "results/Case_00066_0000.nii/PBCT/SVCT-20/DCPnPDP/SVCT_20_0000_0500_100_500_True_nCG-10_None_DDPM/260325_202817"
)

data_dir = Path(
    "results/Case_00066_0000.nii/PBCT/SVCT-20/SITCOM/SVCT_20_0000_0500_100_50_True_nCG-10_None_DDPM/260325_203533"
)

recon_path = data_dir / "recon.nii"
# recon_path = data_dir / "recon_consistent.nii"

print(recon_path)

gt_path = data_dir / "GT.nii"

recon = sitk.ReadImage(str(recon_path))
gt = sitk.ReadImage(str(gt_path))

recon = sitk.GetArrayFromImage(recon)
gt = sitk.GetArrayFromImage(gt)

masks = []
# for i in range(gt.shape[0]):
#     mask, _ = DPER_utils.gantry_removal(gt[i], -0.5)
#     masks.append(mask)

# mask = np.stack(masks, axis=0)  # shape = (192, 192, 192)
# mask = torch.from_numpy(mask).to("cuda").unsqueeze(1)
# print(mask.shape)

recon = torch.tensor(recon).unsqueeze(1).float().to("cuda")
gt = torch.tensor(gt).unsqueeze(1).float().to("cuda")

# recon[mask == 0] = -1  # 将 mask 为 0 的区域设置为 -1
# gt[mask == 0] = -1  # 将 GT 图像中 mask 为

# recon = recon[10:-10]
# gt = gt[10:-10]


print(f"Reconstruction shape: {recon.shape}")
print(f"Ground truth shape: {gt.shape}")


metrics_3D = utils.result.cal_metrics(recon, gt, save_path=data_dir / f"metrics_{recon_path.stem}")
print(f"PSNR: {metrics_3D[0]:.4f}, SSIM: {metrics_3D[1]:.4f}")


metrics_2D_slices = []

for i in range(recon.shape[0]):
    recon_slice = recon[i : i + 1]  # 保持通道维度
    gt_slice = gt[i : i + 1]  # 保持通道维度

    psnr, ssim = utils.result.cal_metrics(
        recon_slice, gt_slice, save_path=data_dir / f"metrics_{recon_path.stem}" / f"slice_{i:03d}"
    )
    metrics_2D_slices.append((psnr, ssim))
    print(f"Slice {i}: PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

print(f"Average PSNR across slices: {np.mean([m[0] for m in metrics_2D_slices]):.4f}")
print(f"Average SSIM across slices: {np.mean([m[1] for m in metrics_2D_slices]):.4f}")

# metrics = utils.result.compute_slice_metrics(recon, gt, data_range=gt.max() - gt.min(), use_lpips=True)
# metrics = DPER_utils.compute_slice_metrics_batch(recon, gt, data_range=gt.max() - gt.min())
metrics = utils.result.compute_slice_metrics_optimized(recon, gt, data_range=gt.max() - gt.min())
utils.result.print_slice_metrics(metrics)

print(metrics["axial"]["PSNR_mean"], metrics["axial"]["SSIM_mean"], metrics["axial"]["LPIPS_mean"])
print(metrics["coronal"]["PSNR_mean"], metrics["coronal"]["SSIM_mean"], metrics["coronal"]["LPIPS_mean"])
print(metrics["sagittal"]["PSNR_mean"], metrics["sagittal"]["SSIM_mean"], metrics["sagittal"]["LPIPS_mean"])
# print(metrics["axial"]["PSNR_mean"], metrics["axial"]["SSIM_mean"])
# print(metrics["coronal"]["PSNR_mean"], metrics["coronal"]["SSIM_mean"])
# print(metrics["sagittal"]["PSNR_mean"], metrics["sagittal"]["SSIM_mean"])

print(f"{metrics['axial']['PSNR_mean']},{metrics['axial']['SSIM_mean']},{metrics['axial']['LPIPS_mean']}")
print(f"{metrics['coronal']['PSNR_mean']},{metrics['coronal']['SSIM_mean']},{metrics['coronal']['LPIPS_mean']}")
print(f"{metrics['sagittal']['PSNR_mean']},{metrics['sagittal']['SSIM_mean']},{metrics['sagittal']['LPIPS_mean']}")

# metrics 写入 yaml 文件

metrics_dict = {
    "PSNR": metrics_3D[0],
    "SSIM": metrics_3D[1],
    "PSNR_axial": metrics["axial"]["PSNR_mean"],
    "SSIM_axial": metrics["axial"]["SSIM_mean"],
    "PSNR_coronal": metrics["coronal"]["PSNR_mean"],
    "SSIM_coronal": metrics["coronal"]["SSIM_mean"],
    "PSNR_sagittal": metrics["sagittal"]["PSNR_mean"],
    "SSIM_sagittal": metrics["sagittal"]["SSIM_mean"],
}

metrics_yaml_path = data_dir / "metrics.yaml"
with open(metrics_yaml_path, "w") as f:
    yaml.dump(metrics_dict, f, default_flow_style=False)
