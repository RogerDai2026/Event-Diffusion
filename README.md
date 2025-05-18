```markdown
# Monocular Depth Estimation Using Residual Diffusion

## 🚀 Overview

This repository implements a two-stage **monocular depth estimation** pipeline based on **residual diffusion modeling**, inspired by NVIDIA’s CorrDiff framework:

1. **Regression UNet**  
   A high-capacity U-Net backbone (`SongUNet` / `SongUNetPosEmbd`) is trained with a simple ℓ₂ (MSE) loss to predict the **conditional mean** depth map from an event-based input.

2. **Residual Diffusion UNet**  
   A second diffusion-based U-Net learns to model the **residual** (fine-scale detail and uncertainty) on top of the regression mean. The two outputs are summed at inference time to produce sharper, more realistic depth estimates.

---

## 📖 Background

Classic monocular depth estimation often struggles with fine details and uncertainty quantification. Residual diffusion modeling:

- **Decouples** the coarse mean prediction (learned with MSE) from the stochastic high-frequency residual.  
- **Leverages** denoising score matching to learn a diffusion process over the residuals.  
- **Improves** both quantitative metrics (MSE, RMSE, abs_rel, sq_rel, δ-accuracy) and visual fidelity of predicted depth maps.

---

## 🏗️ Project Structure

```

.
├── configs/
│   ├── baseline\_regression.yaml       # Regression-only training
│   └── baseline\_diffusion.yaml        # Two-stage CorrDiff training
├── src/
│   ├── models/
│   │   └── corrdiff\_unet.py           # LightningModules for regression & diffusion
│   ├── utils/
│   │   ├── corr\_diff\_utils/           # ResLoss, inference helpers
│   │   └── callbacks/                 # WandB logging, sample visualization
│   └── train.py                       # Entry point for training & resuming
├── data/                              # Dataloaders & example datasets
└── README.md                          # This file

````

---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/monocular-depth-residual-diffusion.git
   cd monocular-depth-residual-diffusion
````

2. **Create a conda environment**

   ```bash
   conda create -n depth-diffusion python=3.10
   conda activate depth-diffusion
   pip install -r requirements.txt
   ```

3. **Prepare your data**

   * Place your event-frame → depth pairs in `data/` following the example structure.
   * Update `configs/data.yaml` if you use a custom dataset.

---

## 🚄 Quick Start

### 1. Train the regression UNet (mean predictor)

```bash
python src/train.py \
  experiment=train_baseline_regression \
  trainer.max_epochs=30 \
  model=baseline_regression
```

### 2. Train the residual diffusion UNet

```bash
python src/train.py \
  experiment=train_baseline_diffusion \
  trainer.max_epochs=50 \
  model=baseline_diffusion \
  model.regression_net_ckpt=logs/train_baseline_regression/version_0/checkpoints/last.ckpt
```

### 3. Resume from checkpoint (same W\&B run)

```bash
python src/train.py \
  trainer.ckpt_path=logs/train_baseline_diffusion/version_0/checkpoints/last.ckpt \
  +logger.wandb.id=YOUR_RUN_ID \
  +logger.wandb.resume=true
```

---

## 🔧 Configuration

All hyperparameters and model choices are exposed via Hydra configs:

* **`configs/baseline_regression.yaml`**
  Regression-only U-Net settings (learning rate, architecture, loss).

* **`configs/baseline_diffusion.yaml`**
  Two-stage CorrDiff settings:

  * `net`: diffusion U-Net wrapper
  * `regression_model_cfg`: regression U-Net constructor args
  * `criterion`: `ResLoss` hyperparameters (patch sizes, σ schedule)
  * `sampling`: sampler settings (overlap, boundary, steps)

---

## 📋 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 🤝 Contributing

We welcome issues and PRs! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 🧑‍💻 Contact

**Roger Dai** ([qdai@uw.edu](mailto:qdai@uw.edu))
**GitHub**: [https://github.com/your-org/monocular-depth-residual-diffusion](https://github.com/your-org/monocular-depth-residual-diffusion)

```
```
