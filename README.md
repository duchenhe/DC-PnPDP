# Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction 🚀

Official PyTorch implementation for "Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction" ✨

[![arXiv](https://img.shields.io/badge/Paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/2602.04162)

## 📖 Overview

Plug-and-Play diffusion prior (PnPDP) frameworks have emerged as a powerful paradigm for solving imaging inverse problems by treating pretrained generative models as modular priors. However, we identify a critical flaw in prevailing PnP solvers (e.g., based on HQS or Proximal Gradient): they function as memoryless operators, updating estimates solely based on instantaneous gradients. This lack of historical tracking inevitably leads to non-vanishing steady-state bias, where the reconstruction fails to strictly satisfy physical measurements under heavy corruption. To resolve this, we propose Dual-Coupled PnP Diffusion, which restores the classical dual variable to provide integral feedback, theoretically guaranteeing asymptotic convergence to the exact data manifold. However, this rigorous geometric coupling introduces a secondary challenge: the accumulated dual residuals exhibit spectrally colored, structured artifacts that violate the Additive White Gaussian Noise (AWGN) assumption of diffusion priors, causing severe hallucinations. To bridge this gap, we introduce Spectral Homogenization (SH), a frequency-domain adaptation mechanism that modulates these structured residuals into statistically compliant pseudo-AWGN inputs. This effectively aligns the solver's rigorous optimization trajectory with the denoiser's valid statistical manifold. Extensive experiments on CT and MRI reconstruction demonstrate that our approach resolves the bias-hallucination trade-off, achieving state-of-the-art fidelity with significantly accelerated convergence.

## 📦 Project Structure

```
.
├─ algorithms/              # DiffPIR, DCPnPDP, and base sampling logic
├─ physics/                 # CT forward model (PBCT) and Radon operators
├─ utils/                   # I/O, metrics, scheduling, argument parsing
├─ recon_PBCT_optimized.py  # Main reconstruction entry point
├─ recon_PBCT.py            # Legacy / reference entry point
├─ recon_PBCT.sh            # Example launch script
└─ results/                 # Example outputs (safe to delete)
```

## 🚀 Quick Start

1) **Prepare data**

Provide a 3D CT volume in NIfTI format (`.nii` or `.nii.gz`).

1) **Run reconstruction**

```bash
bash recon_PBCT.sh
```

Or call the Python entry directly:

```bash
python recon_PBCT.py \
  --method DCPnPDP \
  --task SVCT \
  --degree 20 \
  --gpu 0 \
  --data /path/to/volume.nii.gz \
  --slice-begin 0 \
  --slice-end 500 \
  --slice-step 10 \
  --recon-size 256 \
  --NFE 50 \
  --num-cg 50 \
  --noise-control None \
  --renoise-method DDPM \
  --sigma-max 2 \
  --checkpoint-path /path/to/network-snapshot.pkl \
  --save_dir ./results/
```

<!-- ## 📜 Citation

If you find our work interesting, please consider citing:

```
@article{DCPnPDP2026,
  title   = {Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction},
  author  = {First Author and Second Author and Others},
  journal = {Journal / Conference},
  year    = {2026}
}
``` -->

## 🔐 License

See the LICENSE file for details.
