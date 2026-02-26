import math

import numpy as np
import torch


def batchfy(tensor, batch_size):
    n = len(tensor)
    num_batches = math.ceil(n / batch_size)
    return tensor.chunk(num_batches, dim=0)


def cg_uni(A_fn, b, x=None, rho=0.0, maxiter=50, tol=1e-5):
    if x is None:
        x = torch.zeros_like(b)

    # r = b - (A+rhoI)x
    r = b - A_fn(x, rho)
    p = r.clone()

    def dot(u, v):
        # return (u * v).sum()
        return torch.sum(u.conj() * v)

    rs_old = dot(r, r)

    for _ in range(maxiter):
        Ap = A_fn(p, rho)
        denom = dot(p, Ap)

        if torch.abs(denom) < 1e-30:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = dot(r, r)

        if rs_new < tol**2:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


@torch.no_grad()
def sirt_recon(
    y,  # measured projections
    A,
    AT,  # callables: A(x)->proj, AT(p)->vol
    x0=None,
    n_iter=50,
    lam=1.0,
    eps=1e-6,
    clip=None,  # e.g. (0.0, 1.0) or None
):
    """
    SIRT: x_{k+1} = x_k + lam * D^{-1} A^T M^{-1} (y - A x_k)

    A : forward projector
    AT: back-projector (adjoint)
    """

    # init x
    if x0 is None:
        # you need to know output volume shape; easiest is AT(zeros_like(y))
        x = AT(torch.zeros_like(y))
    else:
        x = x0.clone()

    # --- precompute normalization weights ---
    ones_x = torch.ones_like(x)
    w = A(ones_x)  # proj-domain weights
    w_inv = 1.0 / (w + eps)

    ones_y = torch.ones_like(y)
    v = AT(ones_y)  # vol-domain weights
    v_inv = 1.0 / (v + eps)

    for _ in range(n_iter):
        y_hat = A(x)
        r = y - y_hat
        r_norm = r * w_inv  # M^{-1} r
        g = AT(r_norm)
        dx = g * v_inv  # D^{-1} g
        x = x + lam * dx

        if clip is not None:
            x = x.clamp(clip[0], clip[1])

    return x
