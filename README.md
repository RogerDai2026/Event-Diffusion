# Depth Estimation with an Event Camera (Cross-Modality Autoencoder + PoE)

This project studies **monocular depth estimation from event camera streams** (**events → depth**).  
Event cameras output **asynchronous per-pixel brightness changes (“events”)** rather than intensity frames, which makes standard vision pipelines difficult. Meanwhile, **dense and reliable depth supervision is scarce**. To address these challenges, we build a **cross-modality autoencoder** that aligns **events** and **depth** in a **shared latent space**, enabling training on both **paired** and **unpaired** data via **weak supervision**. We also include a **teacher–student distillation** component to generate dense pseudo-labels when ground truth is incomplete.

---

## TL;DR (What we built)

- **Two modality-specific VAEs** (events VAE + depth VAE) mapped into a **shared, geometry-aware latent space** with ~**4× compression**.
- **Product-of-Experts (PoE)** fusion combines event/depth encoder posteriors when both are available; falls back to unimodal when one modality is missing.
- **Event branch** uses a **weighted NLL-style loss** + **learnable log-variance** to respect event sparsity.
- **Depth branch** leverages a **Marigold-compatible pretrained VAE** for stable depth encoding/decoding.
- Designed as a **front end for latent-space diffusion** (U-Net denoiser operates in the aligned latent space).

---
## Highlights (Depth Prediction)
**Synthetic**

   <img width="900" alt="synthetic_depth_pred" src="https://github.com/user-attachments/assets/8365de5c-d20e-4616-8f24-b367e31004a3" />

**Real (DSEC)**

   <img width="900" alt="real_depth_pred" src="https://github.com/user-attachments/assets/ac612043-9cd9-4119-9c7c-66abef52d20d" />

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
- **Event encoder/decoder:** learns to reconstruct sparse event tensors.
- **Depth encoder/decoder:** uses a pretrained VAE (compatible with latent diffusion depth pipelines).
- **PoE fusion:** merges encoder posteriors into a single latent distribution when both modalities exist.

### Pipeline sketch
<img width="863" height="565" alt="pipeline (1)" src="https://github.com/user-attachments/assets/92c380dd-cd26-4bd2-b7dd-bdd959f1821b" />

### Product-of-Experts fusion (PoE)
When both modalities are present, PoE combines Gaussian posteriors by adding precisions; when missing, it defaults to the available modality posterior.

<img width="1456" height="600" alt="poe" src="https://github.com/user-attachments/assets/4bff4e84-6687-4f7e-ba66-0ab5b7d7d008" />

---

## PoE Reconstruction Results

Reconstruction quality reflects how well the PoE framework learns a shared latent representation of **events** and **depth**.

<details>
<summary><b>MVSEC Reconstructions</b></summary>

**Event reconstruction**  
<img width="1500" height="600" alt="mvsec_event" src="https://github.com/user-attachments/assets/8461934b-98ee-40e8-8fda-b5d033468f8f" />

**Depth reconstruction**  
<img width="2400" height="600" alt="mvsec_depth" src="https://github.com/user-attachments/assets/a906aeb3-0921-41db-b997-381b3ae59d49" />

</details>

<details>
<summary><b>DSEC Reconstructions</b></summary>

**Event reconstruction**  
<img width="1500" height="600" alt="dsec_fintune_event" src="https://github.com/user-attachments/assets/c7e7c353-e5f6-4cda-87d0-6c1c93b44ad0" />

**Depth reconstruction**  
<img width="2400" height="600" alt="dsec_depth" src="https://github.com/user-attachments/assets/ef435c9c-5a63-43b4-a3e0-55079049ff56" />

</details>

---

## Teacher–student distillation (dense pseudo-depth)

We use a pretrained **RGB → depth** model to generate **dense pseudo-labels** aligned to available depth where possible. This helps supervision when ground truth is sparse or contains NaNs.

<img width="2400" height="600" alt="distillation" src="https://github.com/user-attachments/assets/fcce3cae-5c90-41a7-b73e-268d1b9bba51" />

---

## Datasets

- **Synthetic CARLA:** dense depth + controllable scenes (simulated event streams).
- **MVSEC (real):** event + depth sequences in indoor/outdoor settings.
- **DSEC (real):** driving sequences, larger resolution; depth derived from LiDAR disparity.

---

## Depth Prediction (Latent-space Diffusion)

After learning a shared latent space for events and depth, we use a **U-Net denoiser** to perform **latent-space generation** and decode the predicted latent back to a depth map. This addresses:
- **Input-size constraints** of pixel-space diffusion (operate on compressed latents)
- **Alignment issues** by denoising in a modality-aligned latent space

### Current results on synthetic and real datasets

<details>
<summary><b>Synthetic</b></summary>

<img width="5600" height="1800" alt="synthetic_depth_pred" src="https://github.com/user-attachments/assets/8365de5c-d20e-4616-8f24-b367e31004a3" />

</details>

<details>
<summary><b>Real (DSEC)</b></summary>

<img width="5600" height="1800" alt="real_depth_pred" src="https://github.com/user-attachments/assets/ac612043-9cd9-4119-9c7c-66abef52d20d" />

</details>

---

## Reproducibility

- **Random seeds** are fixed for Python / NumPy / PyTorch.
- **Hydra config logging:** each run records the full resolved configuration (model, data, losses, training).
- **Checkpointing:** the best validation checkpoint is saved via PyTorch Lightning.
- **Experiment tracking:** runs are logged to Weights & Biases when enabled.
- **Metric scripts:** evaluation and plotting scripts are stored alongside training code.

**Hardware:** NVIDIA A100 GPU with 80GB VRAM.

---

## Installation

### Requirements
- Python 3.9+ (recommended 3.10)
- CUDA-capable GPU recommended
- PyTorch + PyTorch Lightning
- Hydra
- Weights & Biases (optional)

### Dataset paths
Dataset root paths are configured via Hydra (e.g., `configs/data/*.yaml`) rather than hard-coded in scripts.
