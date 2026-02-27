# Pretrained Checkpoints

We release pretrained diffusion model weights for reproducing and extending the experiments in our paper. This document covers the download link, checkpoint format, training data, and training configuration.

## Checkpoint

We release an **unconditional diffusion model** trained on over 100,000 abdominal CT slices using the [NVLabs EDM](https://github.com/NVlabs/edm) framework. The checkpoint follows EDM's standard serialization format (`EDMPrecond` wrapper) and can be directly loaded via `recon_PBCT.py`.

| Source | Link |
|--------|------|
| Google Drive | [Pretrained checkpoint folder](https://drive.google.com/drive/folders/14YKIpl1x1w19WuF258XeiLLGnZiPSM9y?usp=sharing) |

> If you wish to train a model on your own data, please refer to the [EDM repository](https://github.com/NVlabs/edm) for training scripts and instructions.

## Training Data

**Dataset:** [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)

We selected scans with slice thickness $\leq 1\,\text{mm}$, yielding a high-resolution subset of **225 scans**. The exact case lists used in our experiments are provided in `data/`:

| Split | File | Cases | Slices |
|-------|------|------:|------:|
| Train | [`checkpoints/selected_files_train.csv`](./checkpoints/selected_files_train.csv) | 219 | 104,534 |
| Test  | [`checkpoints/selected_files_valid.csv`](./checkpoints/selected_files_valid.csv) | 6 | — |

**Intensity preprocessing**

- HU clipping: $[-800,\, 800]$
- Linear normalization to $[-1,\, 1]$

**PBCT forward model** (used in reconstruction experiments)

- Geometry: parallel-beam CT (PBCT), 1D flat detector
- Detector bins: 363 &nbsp;|&nbsp; Detector pitch: 1.0 mm

## Training Configuration

Full hyperparameters are recorded in [`checkpoints/training_options.json`](./checkpoints/training_options.json). A concise summary is provided below.

| Parameter | Value |
|-----------|-------|
| Framework | EDM (Karras et al., 2022) |
| Backbone | DDPM++ U-Net (`SongUNet` / `EDMPrecond`) |
| Resolution | 256 × 256 |
| Training mode | Unconditional |
| Batch size | 112 |
| Optimizer | Adam ($\text{lr}=10^{-4}$, $\beta=(0.9,\,0.999)$, $\varepsilon=10^{-8}$) |
| Hardware | 8 × NVIDIA A100 (80 GB) |
| Training time | ~24 hours |
