# Depth Estimation with an Event Camera (Cross-Modality Autoencoder + PoE)

This project studies **monocular depth estimation from event camera streams** (events → depth).  
Event cameras output **asynchronous per-pixel brightness changes (“events”)** rather than intensity frames, which makes standard vision pipelines difficult. Meanwhile, **dense and reliable depth supervision is scarce**. To address these challenges, we build a **cross-modality autoencoder** that aligns **events** and **depth** in a **shared latent space**, enabling training on both **paired** and **unpaired** data via **weak supervision**. We also include a **teacher–student distillation** component to generate dense pseudo-labels when ground truth is incomplete.

> Reference write-up: *Stats450_Project.pdf* (paper-style report for this project).  

---

## TL;DR (What we built)

- **Two modality-specific VAEs** (events VAE + depth VAE) mapped into a **shared, geometry-aware latent space** with ~**4× compression**.
- **Product-of-Experts (PoE)** fusion combines event/depth encoder posteriors when both are available; falls back to unimodal when one modality is missing.
- **Event branch** uses a **weighted NLL-style loss** + **learnable log-variance** to respect event sparsity.
- **Depth branch** leverages a **Marigold-compatible pretrained VAE** for stable depth encoding/decoding.
- Designed as a **front end for latent-space diffusion** (U-Net denoiser operates in the aligned latent space).

---

## Key challenges we target

1. **Input size restriction**  
   Pixel-space diffusion often assumes ~256×256 inputs, but real event datasets can be larger (e.g., DSEC depth at 640×480).  
   → We **diffuse in latent space** (compressed grids).

2. **Data scarcity / incomplete depth labels**  
   Paired event–depth data is limited and depth maps often contain NaNs / invalid regions.  
   → We train with **PoE + weak supervision** on paired and unimodal samples, and use **distillation** for denser supervision.

---

## Method overview

### Cross-modality autoencoder (shared latent)
- Event encoder/decoder: learns to reconstruct sparse event tensors.
- Depth encoder/decoder: uses pretrained VAE (compatible with latent diffusion depth pipelines).
- PoE fusion: merges encoder posteriors into a single latent distribution when both modalities exist.

**Pipeline sketch (add your figure here):**
- **[TODO: insert overview figure]**
  - Path suggestion: `assets/pipeline_overview.png`
  - Markdown:
    ```text
    ![Method overview](assets/pipeline_overview.png)
    ```

### Product-of-Experts fusion (PoE)
When both modalities are present, PoE combines Gaussian posteriors by adding precisions; when missing, it defaults to the available modality posterior.

- **[TODO: insert PoE fusion diagram]**
  - Path suggestion: `assets/poe_fusion.png`

### Teacher–student distillation (dense pseudo-depth)
We use a pretrained RGB→depth model to generate **dense pseudo-labels** aligned to available depth where possible. This helps supervision where ground truth is sparse or contains NaNs.

- **[TODO: insert distillation depth examples]**
  - Path suggestion: `assets/distillation_depth.png`

---

## Datasets

We use:
- **Synthetic CARLA**: dense depth + controllable scenes (simulated event streams).
- **MVSEC (real)**: event + depth sequences in indoor/outdoor settings.
- **DSEC (real)**: driving sequences, larger resolution; depth derived from LiDAR disparity.

**[TODO: insert dataset example montage]**
- Path suggestion: `assets/datasets.png`

---

## Depth masking for real data

Real depth/disparity maps include invalid values (NaNs / out-of-range) due to registration and sensor artifacts.  
We construct a boolean validity mask `V`, set invalid pixels to 0 before normalization, and compute losses/metrics **only on valid pixels**.

---

## Repository structure (suggested)

```text
.
├── configs/                 # Hydra configs (datasets, models, losses, training)
├── data/                    # dataset loaders / preprocessing scripts
├── models/                  # VAE modules, PoE fusion, denoiser (UNet), etc.
├── training/                # Lightning trainers / callbacks / logging
├── scripts/                 # train/eval entrypoints
├── assets/                  # figures for README (placeholders)
└── README.md
