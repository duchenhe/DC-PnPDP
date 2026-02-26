# Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction 🚀

Official PyTorch implementation for "Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction" ✨

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

## 📜 Citation

If you find our work interesting, please consider citing:

```
@article{DCPnPDP2026,
  title   = {Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction},
  author  = {First Author and Second Author and Others},
  journal = {Journal / Conference},
  year    = {2026}
}
```

## 🔐 License

See the LICENSE file for details.