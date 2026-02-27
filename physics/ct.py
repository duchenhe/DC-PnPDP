import torch
import numpy as np

import torch_radon
from utils.data import center_pad_nd, center_unpad_nd

PI = 4 * torch.ones(1).atan()


def _to_internal_cbhw(x: torch.Tensor) -> torch.Tensor:
    """Normalize 4D image layout to [C,B,H,W] for projector internals.

    External callers should use [B,C,H,W]. For backward compatibility, we still
    accept legacy [C,B,H,W] tensors.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape={tuple(x.shape)}")

    # Preferred external layout: [B, 1, H, W]
    if x.shape[1] == 1:
        return x.transpose(0, 1)

    # Legacy layout: [1, B, H, W]
    if x.shape[0] == 1:
        return x

    raise ValueError(f"Ambiguous tensor layout {tuple(x.shape)}. Expected [B,1,H,W] (preferred) or [1,B,H,W] (legacy).")


def _to_external_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert internal [C,B,H,W] layout back to external [B,C,H,W]."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape={tuple(x.shape)}")
    return x.transpose(0, 1)


class PBCT_carterbox:
    def __init__(self, det_count, view_available, view_full_num, recon_size):
        """

        Args:
            det_count (_type_): Number of rays to be projected. The number of rays projected per view.
            view_available (_type_): Number of visible views (if limited-view CT is involved)
            view_full_num (_type_): Total number of views
        """
        self.det_count = det_count
        self.angles_FV = np.linspace(0, np.pi, view_full_num, endpoint=False)
        self.angles_LV = self.angles_FV[view_available]
        self.recon_size = recon_size
        self.view_full_num = view_full_num
        self.LV_list = view_available

        self.radon_FV = torch_radon.ParallelBeam(det_count=det_count, angles=self.angles_FV)
        self.radon_LV = torch_radon.ParallelBeam(det_count=det_count, angles=self.angles_LV)

    # * Main
    # A: Radon Transform
    def A(self, x):
        x = _to_internal_cbhw(x)
        if x.shape[2] != self.det_count:
            x = center_pad_nd(x, (self.det_count, self.det_count), value=0.0).float()

        sino = self.radon_LV.forward(x)
        return _to_external_bchw(sino)

    # A_T
    def A_T(self, y):
        y = _to_internal_cbhw(y)
        recon = self.radon_LV.backward(y)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        norm_factor = PI.item() / (2 * len(self.angles_LV))
        recon = recon * norm_factor

        return _to_external_bchw(recon)

    # Moore–Penrose （pseudo-inverse）: FBP
    def A_dagger(self, y, filter_name="ramp"):
        y = _to_internal_cbhw(y)
        sino_filtered = self.radon_LV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_LV.backward(sino_filtered)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        return _to_external_bchw(recon)

    # * ------------------------------------------

    def A_FV(self, x):
        x = _to_internal_cbhw(x)
        if x.shape[2] != self.det_count:
            x = center_pad_nd(x, (self.det_count, self.det_count), value=0.0)

        sino = self.radon_FV.forward(x)
        return _to_external_bchw(sino)

    def FBP_FV(self, y, filter_name="ramp"):
        y = _to_internal_cbhw(y)
        sino_filtered = self.radon_FV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_FV.backward(sino_filtered)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        return _to_external_bchw(recon)

    def BP_FV(self, y):
        y = _to_internal_cbhw(y)
        recon = self.radon_FV.backward(y)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        norm_factor = PI.item() / (2 * len(self.angles_FV))
        recon = recon * norm_factor

        return _to_external_bchw(recon)

    def get_angles(self):
        return self.angles_FV, self.angles_LV


class FBCT_carterbox:
    def __init__(self, det_count, view_available, view_full_num, recon_size):
        """

        Args:
            det_count (_type_): Number of rays to be projected. The number of rays projected per view.
            view_available (_type_): Number of visible views (if limited-view CT is involved)
            view_full_num (_type_): Total number of views
        """
        self.det_count = det_count
        self.angles_FV = np.linspace(0, np.pi, view_full_num, endpoint=False)
        self.angles_LV = self.angles_FV[view_available]
        self.recon_size = recon_size
        self.view_full_num = view_full_num
        self.LV_list = view_available

        self.radon_FV = torch_radon.FanBeam(det_count=det_count, angles=self.angles_FV)
        self.radon_LV = torch_radon.FanBeam(det_count=det_count, angles=self.angles_LV)

    # * Main
    # A: Radon Transform
    def A(self, x):
        x = _to_internal_cbhw(x)
        if x.shape[2] != self.det_count:
            x = center_pad_nd(x, (self.det_count, self.det_count), value=0.0).float()

        sino = self.radon_LV.forward(x)
        return _to_external_bchw(sino)

    # A_T
    def A_T(self, y):
        y = _to_internal_cbhw(y)
        recon = self.radon_LV.backward(y)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        norm_factor = PI.item() / (2 * len(self.angles_LV))
        recon = recon * norm_factor

        return _to_external_bchw(recon)

    # Moore–Penrose （pseudo-inverse）: FBP
    def A_dagger(self, y, filter_name="ramp"):
        y = _to_internal_cbhw(y)
        sino_filtered = self.radon_LV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_LV.backward(sino_filtered)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        return _to_external_bchw(recon)

    # * ------------------------------------------

    def A_FV(self, x):
        x = _to_internal_cbhw(x)
        if x.shape[2] != self.det_count:
            x = center_pad_nd(x, (self.det_count, self.det_count), value=0.0)

        sino = self.radon_FV.forward(x)
        return _to_external_bchw(sino)

    def FBP_FV(self, y, filter_name="ramp"):
        y = _to_internal_cbhw(y)
        sino_filtered = self.radon_FV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_FV.backward(sino_filtered)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        return _to_external_bchw(recon)

    def BP_FV(self, y):
        y = _to_internal_cbhw(y)
        recon = self.radon_FV.backward(y)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        norm_factor = PI.item() / (2 * len(self.angles_FV))
        recon = recon * norm_factor

        return _to_external_bchw(recon)

    def get_angles(self):
        return self.angles_FV, self.angles_LV


class CBCT_carterbox:
    def __init__(
        self,
        angles_FV,
        angle_LV,
        det_count_u,
        det_count_v,
        det_spacing_u,
        det_spacing_v,
        src_dist=300,
        det_dist=300,
        pitch=0,
        base_z=0,
        volume=None,
    ):
        """

        Args:
            det_count (_type_): Number of rays to be projected. The number of rays projected per view.
            view_available (_type_): Number of visible views (if limited-view CT is involved)
            view_full_num (_type_): Total number of views
        """
        self.angles_FV = angles_FV
        self.angles_LV = angle_LV

        self.det_count_u = det_count_u
        self.det_count_v = det_count_v
        self.det_spacing_u = det_spacing_u
        self.det_spacing_v = det_spacing_v
        self.src_dist = src_dist
        self.det_dist = det_dist
        self.volume = volume

        self.radon_FV = torch_radon.ConeBeam(
            det_count_u=self.det_count_u,
            angles=self.angles_FV,
            src_dist=self.src_dist,
            det_dist=self.det_dist,
            det_count_v=self.det_count_v,
            det_spacing_u=self.det_spacing_u,
            det_spacing_v=self.det_spacing_v,
            pitch=pitch,
            base_z=base_z,
            volume=self.volume,
        )

        self.radon_LV = torch_radon.ConeBeam(
            det_count_u=self.det_count_u,
            angles=self.angles_LV,
            src_dist=self.src_dist,
            det_dist=self.det_dist,
            det_count_v=self.det_count_v,
            det_spacing_u=self.det_spacing_u,
            det_spacing_v=self.det_spacing_v,
            pitch=pitch,
            base_z=base_z,
            volume=self.volume,
        )

    def A(self, x):
        x = _to_internal_cbhw(x)
        sino = self.radon_LV.forward(x)
        return _to_external_bchw(sino)

    def A_T(self, y):
        y = _to_internal_cbhw(y)
        recon = self.radon_LV.backward(y)
        return _to_external_bchw(recon)

    def A_FV(self, x: torch.Tensor):
        x = _to_internal_cbhw(x)
        sino = self.radon_FV.forward(x)
        return _to_external_bchw(sino)

    def BP_FV(self, y):
        y = _to_internal_cbhw(y)
        recon = self.radon_FV.backward(y)
        return _to_external_bchw(recon)

    def get_angles(self):
        return self.angles_FV, self.angles_LV
