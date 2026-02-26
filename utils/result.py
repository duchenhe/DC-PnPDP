from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def cal_metrics(pred, gt, save_path, sitk_info=None):
    """
    Calculate the metrics between the prediction and the ground truth.
    :param pred: The prediction
    :param gt: The ground truth
    :return: The metrics
    """
    psnr = peak_signal_noise_ratio(pred, gt, data_range=gt.max() - gt.min())
    psnr = psnr.item()
    ssim = structural_similarity_index_measure(pred, gt, data_range=gt.max() - gt.min())
    ssim = ssim.item()
    error_map = torch.abs(pred - gt)
    if save_path is None:
        return psnr, ssim
    save_nii_image(error_map, save_path / "error_map.nii.gz", sitk_info=sitk_info)
    save_nii_image(pred, save_path / "pred.nii.gz", sitk_info=sitk_info)
    save_nii_image(gt, save_path / "gt.nii.gz", sitk_info=sitk_info)
    # save psnr and ssim to txt file
    with open(save_path / "metrics.txt", "w") as f:
        f.write(f"PSNR: {psnr:.8f}\n")
        f.write(f"SSIM: {ssim:.8f}\n")

    return psnr, ssim


def save_nii_image(image, save_path, sitk_info=None):
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        if image.requires_grad:
            image = image.detach()
        image = image.float().squeeze().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.squeeze()

    sitk_image = sitk.GetImageFromArray(image)
    if sitk_info is not None:
        sitk_image.SetSpacing(sitk_info["spacing"])
        sitk_image.SetDirection(sitk_info["direction"])
    sitk.WriteImage(sitk_image, save_path)


def compute_slice_metrics_optimized(
    pred: torch.Tensor,
    gt: torch.Tensor,
    data_range: float = 1.0,
    lpips_net: torch.nn.Module = None,
    lpips_batch_size: int = 32,
    use_mask: bool = True,
):
    """
    Efficiently compute PSNR, SSIM, and LPIPS metrics for a 3D volume across three axial directions.

    Args:
        pred (torch.Tensor): The predicted volume, with shape [Z, 1, H, W].
        gt (torch.Tensor): The ground truth volume, with shape [Z, 1, H, W].
        data_range (float): The dynamic range of the original pixel values (e.g., 1.0 for 0–1, 255 for 0–255).
        lpips_net (torch.nn.Module, optional):
            A pre-initialized LPIPS network model. If None, LPIPS computation is skipped.

    Returns:
        dict: A dictionary containing metrics for the 'axial', 'coronal', and 'sagittal' directions.
    """

    assert pred.shape == gt.shape, "Volumes must have the same shape"
    assert pred.ndim == 4 and pred.shape[1] == 1, "Expect shape [Z, 1, H, W]"

    def _permute_for_axis(x, axis):
        if axis == "axial":  # Z direction slices -> [Z, 1, H, W]
            return x
        elif axis == "coronal":  # H direction slices -> [H, 1, Z, W]
            return x.permute(2, 1, 0, 3)
        elif axis == "sagittal":  # W direction slices -> [W, 1, Z, H]
            return x.permute(3, 1, 0, 2)
        else:
            raise ValueError("axis must be 'axial', 'coronal' or 'sagittal'")

    results = {}
    for axis in ["axial", "coronal", "sagittal"]:
        pred_slices = _permute_for_axis(pred, axis)
        gt_slices = _permute_for_axis(gt, axis)

        variation = gt_slices.amax(dim=(2, 3), keepdim=True) - gt_slices.amin(dim=(2, 3), keepdim=True)
        valid_mask = (variation > 1e-6).squeeze()

        axis_results = {}

        if valid_mask.sum() == 0:
            axis_results = {
                "PSNR_mean": float("nan"),
                "SSIM_mean": float("nan"),
                "PSNR_each": torch.tensor([]),
                "SSIM_each": torch.tensor([]),
            }
            if lpips_net is not None:
                axis_results.update({"LPIPS_mean": float("nan"), "LPIPS_each": torch.tensor([])})
            results[axis] = axis_results
            continue

        if use_mask:
            valid_pred = pred_slices[valid_mask]
            valid_gt = gt_slices[valid_mask]
        else:
            valid_pred = pred_slices
            valid_gt = gt_slices

        psnr_each = peak_signal_noise_ratio(
            preds=valid_pred, target=valid_gt, data_range=data_range, dim=(1, 2, 3), reduction="none"
        )
        ssim_each = structural_similarity_index_measure(
            preds=valid_pred, target=valid_gt, data_range=data_range, reduction="none"
        )
        axis_results.update(
            {
                "PSNR_mean": psnr_each.mean().item(),
                "SSIM_mean": ssim_each.mean().item(),
                "PSNR_each": psnr_each,
                "SSIM_each": ssim_each,
            }
        )

        if lpips_net is not None:
            lpips_results_list = []
            num_valid_slices = valid_pred.shape[0]

            with torch.no_grad():
                for i in range(0, num_valid_slices, lpips_batch_size):
                    pred_chunk = valid_pred[i : i + lpips_batch_size]
                    gt_chunk = valid_gt[i : i + lpips_batch_size]

                    norm_pred = (pred_chunk / data_range) * 2 - 1
                    norm_gt = (gt_chunk / data_range) * 2 - 1

                    lpips_chunk = lpips_net(norm_pred, norm_gt)
                    lpips_results_list.append(lpips_chunk)

            lpips_each = torch.cat(lpips_results_list).squeeze()

            axis_results.update(
                {
                    "LPIPS_mean": lpips_each.mean().item(),
                    "LPIPS_each": lpips_each,
                }
            )

        results[axis] = axis_results

    return results

def print_slice_metrics(metrics, include_lpips=False):
    if include_lpips:
        print(f"{'View':<10}{'PSNR':>12}{'SSIM':>12}{'LPIPS':>12}")
        print(
            f"{'Axial':<10}{metrics['axial']['PSNR_mean']:12.2f}{metrics['axial']['SSIM_mean']:12.3f}{metrics['axial']['LPIPS_mean']:12.3f}"
        )
        print(
            f"{'Coronal':<10}{metrics['coronal']['PSNR_mean']:12.2f}{metrics['coronal']['SSIM_mean']:12.3f}{metrics['coronal']['LPIPS_mean']:12.3f}"
        )
        print(
            f"{'Sagittal':<10}{metrics['sagittal']['PSNR_mean']:12.2f}{metrics['sagittal']['SSIM_mean']:12.3f}{metrics['sagittal']['LPIPS_mean']:12.3f}"
        )
    else:
        print(f"{'View':<10}{'PSNR':>12}{'SSIM':>12}")
        print(f"{'Axial':<10}{metrics['axial']['PSNR_mean']:12.4f}{metrics['axial']['SSIM_mean']:12.4f}")
        print(f"{'Coronal':<10}{metrics['coronal']['PSNR_mean']:12.4f}{metrics['coronal']['SSIM_mean']:12.4f}")
        print(f"{'Sagittal':<10}{metrics['sagittal']['PSNR_mean']:12.4f}{metrics['sagittal']['SSIM_mean']:12.4f}")