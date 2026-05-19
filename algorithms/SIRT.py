import torch


class SIRTPseudoInverse:
    """
    Use SIRT as a stable approximate pseudo-inverse A^dagger for CT.
    Only requires A (forward) and AT (backprojection/adjoint).
    Volume shape assumed to be [B, D, H, W] = [1, 500, 256, 256] in your case.
    """

    def __init__(self, A, AT, vol_shape, eps=1e-6, device=None, dtype=torch.float32):
        self.A = A
        self.AT = AT
        self.vol_shape = vol_shape
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self._w_inv = None  # projection-domain weights inverse
        self._v_inv = None  # image-domain weights inverse

    @torch.no_grad()
    def _precompute_weights(self, proj_like):
        """
        Precompute:
          w_inv = 1 / (A(ones_x) + eps)
          v_inv = 1 / (AT(ones_y) + eps)
        proj_like: a tensor with the same shape as projection b (e.g., y).
        """
        dev = proj_like.device if self.device is None else self.device
        dt = proj_like.dtype if proj_like.dtype.is_floating_point else self.dtype

        ones_x = torch.ones(self.vol_shape, device=dev, dtype=dt)
        w = self.A(ones_x)  # projection weights
        self._w_inv = 1.0 / (w + self.eps)

        ones_y = torch.ones_like(proj_like, device=dev, dtype=dt)
        v = self.AT(ones_y)  # voxel weights
        self._v_inv = 1.0 / (v + self.eps)

    @torch.no_grad()
    def pinv(self, b, x0=None, n_iter=10, lam=1.0, min_value=None, clip=None):
        """
        Approximate x = A^dagger b via SIRT iterations.

        Args:
          b: projection data (same shape as A(x)).
          x0: initial volume [B, D, H, W]. If None, start from zeros.
          n_iter: SIRT iterations (DDNM里通常用小迭代步数，比如 5~30)
          lam: relaxation
          nonneg: enforce x>=0
          clip: tuple (lo, hi) or None

        Returns:
          x: reconstructed volume
        """
        # lazy init weights
        if (self._w_inv is None) or (self._v_inv is None):
            self._precompute_weights(proj_like=b)

        if x0 is None:
            x = torch.zeros(self.vol_shape, device=b.device, dtype=b.dtype)
        else:
            x = x0.clone()

        w_inv = self._w_inv
        v_inv = self._v_inv

        for _ in range(n_iter):
            r = b - self.A(x)  # projection residual
            r = r * w_inv  # M^{-1} r
            dx = self.AT(r)  # backproject
            dx = dx * v_inv  # D^{-1} dx
            x = x + lam * dx

            if min_value is not None:
                x = torch.clamp_min(x, min_value)
            if clip is not None:
                x = x.clamp(clip[0], clip[1])

        return x
