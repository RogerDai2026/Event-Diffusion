# Depth Estimation with an Event Camera (Cross-Modality Autoencoder + PoE)

This project studies **monocular depth estimation from event camera streams** (events → depth).  
Event cameras output **asynchronous per-pixel brightness changes (“events”)** rather than intensity frames, which makes standard vision pipelines difficult. Meanwhile, **dense and reliable depth supervision is scarce**. To address these challenges, we build a **cross-modality autoencoder** that aligns **events** and **depth** in a **shared latent space**, enabling training on both **paired** and **unpaired** data via **weak supervision**. We also include a **teacher–student distillation** component to generate dense pseudo-labels when ground truth is incomplete.

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

**Pipeline sketch:**
<img width="863" height="565" alt="pipeline (1)" src="https://github.com/user-attachments/assets/92c380dd-cd26-4bd2-b7dd-bdd959f1821b" />

### Product-of-Experts fusion (PoE)
When both modalities are present, PoE combines Gaussian posteriors by adding precisions; when missing, it defaults to the available modality posterior

<img width="1456" height="600" alt="poe" src="https://github.com/user-attachments/assets/4bff4e84-6687-4f7e-ba66-0ab5b7d7d008" />

PoE Reconstruction Result (reconstruction qualities helps to infer how well PoE framework learns the latent representation of event and depth modalities)
MVSEC Event reconstruction
<img width="2400" height="600" alt="mvsec_event" src="https://github.com/user-attachments/assets/8461934b-98ee-40e8-8fda-b5d033468f8f" />

MVSEC Depth reconstruction
<img width="2400" height="600" alt="mvsec_depth" src="https://github.com/user-attachments/assets/a906aeb3-0921-41db-b997-381b3ae59d49" />

DSEC Event reconstruction
<img width="2400" height="600" alt="dsec_fintune_event" src="https://github.com/user-attachments/assets/c7e7c353-e5f6-4cda-87d0-6c1c93b44ad0" />

DSEC Depth reconstruction
<img width="2400" height="600" alt="dsec_depth" src="https://github.com/user-attachments/assets/ef435c9c-5a63-43b4-a3e0-55079049ff56" />


### Teacher–student distillation (dense pseudo-depth)
We use a pretrained RGB→depth model to generate **dense pseudo-labels** aligned to available depth where possible. This helps supervision where ground truth is sparse or contains NaNs.

<img width="694" height="347" alt="Screenshot 2026-01-04 at 4 19 30 PM" src="https://github.com/user-attachments/assets/fcce3cae-5c90-41a7-b73e-268d1b9bba51" />


## Datasets

We use:
- **Synthetic CARLA**: dense depth + controllable scenes (simulated event streams).
- **MVSEC (real)**: event + depth sequences in indoor/outdoor settings.
- **DSEC (real)**: driving sequences, larger resolution; depth derived from LiDAR disparity.

---
## Depth Prediction (Latent-space Diffusion)

After learning a shared latent space for events and depth, we use a U-Net denoiser to perform **latent-space generation** and decode the predicted latent back to a depth map. This is intended to address:
- **Input-size constraints** of pixel-space diffusion (operate on compressed latents)
- **Alignment issues** by denoising in a modality-aligned latent space

### Status
This part of the pipeline is **in progress**.

### Current Results on Synthetic and Real datasets

Synthetic
<img width="5600" height="1800" alt="media_images_val_conditional_samples_3799_3387e36061445b09af40" src="https://github.com/user-attachments/assets/8365de5c-d20e-4616-8f24-b367e31004a3" />

Real(DSEC)
<img width="5600" height="1800" alt="media_images_val_conditional_samples_6671_5b03e28f694c8893aa44" src="https://github.com/user-attachments/assets/ac612043-9cd9-4119-9c7c-66abef52d20d" />

---

## Reproducibility

We follow standard practices for reproducibility:

- **Random seeds** are fixed for Python / NumPy / PyTorch.
- **Hydra config logging:** each run records the full resolved configuration (model, data, losses, training).
- **Checkpointing:** the best validation checkpoint is saved via PyTorch Lightning.
- **Experiment tracking:** runs are logged to Weights & Biases when enabled.
- **Metric scripts:** evaluation and plotting scripts are stored alongside training code.

**Hardware note:** reported experiments were trained on an NVIDIA A100 GPU with 80GB VRAM.

## Installation

### Requirements
- Python 3.9+ (recommended 3.10)
- CUDA-capable GPU recommended (main experiments were run on NVIDIA A100 80GB)
- PyTorch + PyTorch Lightning
- Hydra
- Weights & Biases (optional, for logging)

### Dataset preparation
> Recommended: keep dataset root paths in `configs/data/*.yaml` (Hydra) rather than hardcoding paths.


