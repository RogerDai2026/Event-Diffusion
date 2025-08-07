import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import seaborn as sns  # optional for nicer plots
from tqdm import tqdm

# ----- utility functions ---------------------------------------------------
def compute_mmd(x, y, gamma=1.0):
    def rbf(a, b, gamma):
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(diff ** 2, axis=2))
    xx = rbf(x, x, gamma)
    yy = rbf(y, y, gamma)
    xy = rbf(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def energy_distance(x, y):
    def pairwise_dist(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    exx = pairwise_dist(x, x)
    eyy = pairwise_dist(y, y)
    exy = pairwise_dist(x, y)
    return 2 * exy.mean() - exx.mean() - eyy.mean()

# Online mean/cov estimator (Welford) to avoid full materialization if needed
class OnlineStats:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, x):
        # x: (B, D)
        for xi in x:
            self.n += 1
            delta = xi - self.mean
            self.mean += delta / self.n
            delta2 = xi - self.mean
            self.M2 += np.outer(delta, delta2)

    def finalize(self):
        if self.n < 2:
            cov = np.zeros_like(self.M2)
        else:
            cov = self.M2 / (self.n - 1)
        return self.mean, cov

# ----- load latent codes with progress bar ---------------------------------
def load_latents(latent_dir, exts=(".npy", ".pt", ".pth"), max_files=None):
    latents = []
    paths = []
    for root, _, files in os.walk(latent_dir):
        for fname in files:
            if fname.lower().endswith(exts):
                paths.append(os.path.join(root, fname))
    if max_files:
        paths = paths[:max_files]

    if not paths:
        raise FileNotFoundError(f"No latent files found under {latent_dir} with extensions {exts}")

    for path in tqdm(paths, desc="Loading latent files", unit="file"):
        fname = os.path.basename(path)
        try:
            if fname.lower().endswith(".npy"):
                z = np.load(path)
            else:
                z = torch.load(path, map_location="cpu")
                if isinstance(z, dict):
                    # heuristics: if saved as dict, try common keys
                    for k in ("latent", "z", "code"):
                        if k in z:
                            z = z[k]
                            break
                if hasattr(z, "cpu"):
                    z = z.cpu().numpy()
                else:
                    z = np.array(z)
            if z.ndim == 1:
                latents.append(z[None, :])
            else:
                latents.append(z.reshape(-1, z.shape[-1]))
        except Exception as e:
            print(f"[load_latents] skipped {path}: {e}")

    if not latents:
        raise RuntimeError("Loaded zero latent vectors.")
    latents = np.concatenate(latents, axis=0)
    return latents  # (N, D)

# ----- main inspection -----------------------------------------------------
def inspect_latent(latents, name="latent", sample_size=5000, do_plots=True):
    N, D = latents.shape
    print(f"[{name}] samples: {N}, dim: {D}")

    # subsample for computation if huge
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        z = latents[idx]
    else:
        z = latents

    # 1. Mean and covariance
    mean = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    print(f"Mean norm (should be ~0): {np.linalg.norm(mean):.4f}")
    print(f"Cov diag mean (should be ~1): {np.mean(np.diag(cov)):.4f}, std: {np.std(np.diag(cov)):.4f}")

    # 2. Eigenvalues
    eigs = np.linalg.eigvalsh(cov)
    print(f"Top 5 eigenvalues: {eigs[-5:]}")
    print(f"Bottom 5 eigenvalues: {eigs[:5]}")

    # 3. Skew/kurtosis
    skew = stats.skew(z, axis=0)
    kurt = stats.kurtosis(z, axis=0)
    print(f"Mean skewness: {np.mean(skew):.4f}, mean excess kurtosis: {np.mean(kurt):.4f}")

    if do_plots:
        # 4. QQ plots
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        dims = np.random.choice(D, size=6, replace=False)
        for ax, d in zip(axes.flat, dims):
            stats.probplot(z[:, d], dist="norm", plot=ax)
            ax.set_title(f"Dim {d}")
        fig.suptitle(f"QQ-plots of {name}")
        plt.tight_layout()
        plt.show()

        # 5. Histogram marginal
        plt.figure()
        sns.histplot(z[:, 0], stat="density", label="latent dim0", kde=True)
        x = np.linspace(-4, 4, 400)
        plt.plot(x, stats.norm.pdf(x), label="N(0,1)")
        plt.title(f"{name} marginal vs N(0,1)")
        plt.legend()
        plt.show()

        # 6. PCA projection
        pca = PCA(n_components=2)
        z2 = pca.fit_transform(z)
        plt.figure()
        plt.scatter(z2[:, 0], z2[:, 1], alpha=0.5, s=5)
        plt.title(f"PCA of {name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    # 7. MMD / energy to standard normal
    z_prior = np.random.randn(*z.shape)
    mmd_val = compute_mmd(z, z_prior, gamma=1.0 / D)
    ed = energy_distance(z, z_prior)
    print(f"MMD(z, N(0,I)): {mmd_val:.6f}")
    print(f"Energy distance: {ed:.6f}")

    return {
        "mean": mean,
        "cov": cov,
        "eigs": eigs,
        "skew": skew,
        "kurt": kurt,
        "mmd": mmd_val,
        "energy": ed,
    }


if __name__ == "__main__":
    latent_dir = "/shared/qd8/vae_latent_nbin3/train/latents"
    # load (with progress bar)
    latent_codes = load_latents(latent_dir)
    # inspect
    stats_dict = inspect_latent(latent_codes, name="combined latent", sample_size=500)


# -------------------- usage example ----------------------------------------
# if __name__ == "__main__":
#     latent_dir = "/shared/qd8/vae_latent_nbin3/train/latents"  # adapt if event vs depth separated
#     latent_codes = load_latents(latent_dir)  # (N, D)
#     inspect_latent(latent_codes, name="combined latent")
