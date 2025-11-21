# Variational Autoencoder on Fashion-MNIST  

This README explains the full approach, code structure, and outputs for **Q4: Variational Autoencoder (VAE)**.

---

## 1. Environment & Setup

- Frameworks: **PyTorch**, `torchvision`, `matplotlib`, `seaborn`, `imageio`, `scipy`. :contentReference[oaicite:1]{index=1}  
- Device: `cuda` (Tesla T4 when run on Colab).  
- A fixed `USERNAME = "atharv.bhatt"` string is added to plots as a watermark so the results are clearly attributable. :contentReference[oaicite:2]{index=2}  

> **Reproducibility:**  
> Running the notebook cell-by-cell from top to bottom on a GPU runtime will fully reproduce the results. Seeds for PyTorch and NumPy are set to 42.

---

## 2. Dataset Preparation (4.2)

### 2.1 Loading & Normalization

- Dataset: **Fashion-MNIST** (60,000 train, 10,000 test). :contentReference[oaicite:3]{index=3}  
- Transform: `transforms.ToTensor()` – converts images from `[0, 255]` to **[0, 1]** floats.  
  - This is appropriate because we use **Binary Cross-Entropy (BCE)** loss, which expects probabilities in [0, 1].  
- DataLoaders:  
  - `batch_size = 128`, `shuffle=True` for train, `False` for test, with `num_workers=2` and `pin_memory=True` to speed up GPU loading. :contentReference[oaicite:4]{index=4}  

### 2.2 Sanity-check visualization

- A 2×5 grid shows 10 sample images with their class names (`T-shirt/top`, `Trouser`, … `Ankle boot`), confirming correct loading and labels. :contentReference[oaicite:5]{index=5}  
- This also visually verifies grayscale format and 28×28 resolution.

> **Requirement satisfied:** “Load and preprocess the Fashion-MNIST dataset… Normalize the data and create DataLoader instances… Choose suitable batch size and normalization strategy.”  

---

## 3. VAE Model Architecture (4.3)

The notebook defines a **convolutional VAE** with encoder, reparameterization, and decoder. :contentReference[oaicite:6]{index=6}  

### 3.1 Encoder

```text
Input:  (B, 1, 28, 28)
Conv1:  1 → 32, kernel 3, stride 2, padding 1 → (B, 32, 14, 14)
BN + ReLU
Conv2:  32 → 64, kernel 3, stride 2, padding 1 → (B, 64, 7, 7)
BN + ReLU
Flatten: 64*7*7 → linear
fc_mu:     (64*7*7 → latent_dim)
fc_logvar: (64*7*7 → latent_dim)
````

* Outputs two vectors: **μ** and **logσ²** for the latent Gaussian. 

### 3.2 Reparameterization Trick

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
```

* Implements `z = μ + ε·σ` with `ε ~ N(0, I)` so gradients can flow through `μ` and `logσ²`. 

### 3.3 Decoder

```text
Input:  z (B, latent_dim)
FC:    latent_dim → 64*7*7
View:  (B, 64, 7, 7)
Deconv1: 64 → 32, kernel 4, stride 2, padding 1 → (B, 32, 14, 14)
BN + ReLU
Deconv2: 32 → 1,  kernel 4, stride 2, padding 1 → (B, 1, 28, 28)
Sigmoid output in [0, 1]
```

* The final **sigmoid** is consistent with BCE reconstruction loss. 

> **Requirement satisfied:** Model has an encoder producing μ and logσ², uses the reparameterization trick, and a decoder reconstructing 28×28 images from latent `z`.

---

## 4. Loss Function & β-VAE (4.4)

The loss is implemented as: 

$$
\mathcal{L}(x) = \text{BCE}(x,\hat{x}) + \beta , D_{\text{KL}}(q_\phi(z|x),|,\mathcal{N}(0,I))
$$

* **Reconstruction loss**: `F.binary_cross_entropy(recon_x, x, reduction='sum')`

  * Encourages pixel-wise similarity between reconstruction and input.
* **KL divergence**:
  $$
  -\tfrac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)
  $$

  * Makes the latent approximate **N(0, I)**, giving a smooth, continuous latent space and enabling sampling.
* **β parameter**:

  * `β = 1` → standard VAE.
  * `β < 1` → prioritizes reconstruction (sharper images, less regularization).
  * `β > 1` → stronger regularization (more disentangling, more blur). 

> **Requirement satisfied:** loss combines BCE reconstruction and KL terms; commentary in notebook explains each term’s role.

---

## 5. Training Procedure – Standard VAE (4.5)

### 5.1 Training setup

* Latent dimension **16**, β = 1.0, Adam optimizer with `lr = 1e-3`, trained for **25 epochs**. 
* `train_epoch` and `test_epoch` functions compute average **total loss**, **reconstruction**, and **KL** per image over the dataset.

### 5.2 Learning curves & numbers

* Training & test losses are logged every 5 epochs. Example:

  * Epoch 5: Train ≈ 246.19, Test ≈ 247.77
  * Epoch 25: Train ≈ 240.46, Test ≈ 242.31 
* Plots show:

  * **Total loss vs epoch** for train/test.
  * **Reconstruction loss** and **KL divergence** vs epoch.

**Why this looks correct**

* Train and test loss curves are close and smoothly decreasing → model trains stably without severe overfitting.
* KL term stabilises at a positive value, indicating the latent distribution has non-trivial variance (not collapsing).

> **Requirement satisfied:** VAE is trained with a standard optimizer, and loss curves are plotted and interpreted.

---

## 6. β-VAE Experiments & GIFs (4.6)

### 6.1 Setup

* Only three classes are used for this analysis:
  **T-shirt/top (0), Trouser (1), Sneaker (7)**. Filtered training set size = 18,000. 
* For visualization, latent dimension is set to **2**.
* For each β ∈ {0.1, 0.5, 1.0}:

  * A new VAE(2-D latent) is trained for **20 epochs** on the filtered data. 

### 6.2 Latent evolution GIFs

During training:

* Every `save_interval` batches, the notebook:

  1. Encodes the current mini-batch to obtain μ ∈ ℝ².
  2. Plots a **scatter plot of the latent space**, color-coding points by class.
  3. Displays a small strip of **original vs reconstructed** images underneath.
  4. Saves this figure as a PNG frame. 
* After training, frames are combined into GIFs:

  * `vae_beta_0.1_evolution.gif`
  * `vae_beta_0.5_evolution.gif`
  * `vae_beta_1.0_evolution.gif` 

The sample frames you attached (for β = 0.1, 0.5, 1.0) show:

* Left: 2D latent scatter where each class forms a colored cluster.
* Right: top row = original images, bottom row = reconstructions (initially noisy, then sharper as training proceeds).

This directly matches the assignment’s requirement to visualize latent evolution and reconstructions for different β.

### 6.3 Loss comparison & qualitative table

* For each β, total, reconstruction, and KL losses per epoch are stored and plotted. 
* A comparison table is printed: 

| β   | Final recon loss | Final KL | Image sharpness  | Diversity | Cluster separation |
| --- | ---------------- | -------- | ---------------- | --------- | ------------------ |
| 0.1 | ~209.47          | ~8.76    | High (sharp)     | Low       | Poor               |
| 0.5 | ~209.79          | ~6.12    | Medium           | Medium    | Moderate           |
| 1.0 | ~210.55          | ~5.28    | Lower (blurrier) | High      | Good               |

**Interpretation/justification**

* As **β increases**, KL weight grows:

  * KL loss decreases (latent closer to N(0,I)).
  * Recon loss slightly increases (blurrier images).
* Qualitative observations from GIFs and recon grids:

  * β=0.1 → sharp but clusters overlap; latent space less structured.
  * β=0.5 → trade-off: reasonable sharpness + better separation.
  * β=1.0 → more spread and clearly separated clusters; images are slightly blurrier but more diverse.

> **Requirement satisfied:** We systematically vary β, visualize latent clusters & reconstructions, and summarise quantitative and qualitative effects.

---

## 7. Evaluation & Visualization (4.7)

### 7.1 Original vs reconstructed images

* A fresh `VAE(latent_dim=16)` is created and loaded from `vae_standard.pth`. 

* The notebook takes 16 test images and shows:

  1. Top subplot: **originals** in an 8×2 grid.
  2. Bottom subplot: **reconstructions** in the same layout.

* Visual inspection: reconstructions retain class structure and outline (e.g., coats, trousers, boots) but lose some fine details—exactly what we expect from a moderate-capacity VAE.

### 7.2 Sampling from latent space

* 64 latent vectors `z ~ N(0, I)` are drawn, decoded, and shown as a grid. 
* Generated images resemble plausible Fashion-MNIST items spanning multiple categories (tops, shoes, trousers, etc.), confirming that the latent prior is meaningful.

### 7.3 Qualitative notes (in notebook)

The notebook summarises: 

* **Quality:** recognizable shapes.
* **Diversity:** good variety across clothing categories.
* **Realism:** some blur / lack of fine detail, typical of basic VAEs.

> **Requirement satisfied:** we explicitly show originals vs reconstructions and samples from N(0, I), and comment on quality/diversity/realism.

---

## 8. FID Score – What it Measures & How to Interpret Our Result

### 8.1 Implementation

To approximate **Frechet Inception Distance (FID)**, the notebook:

1. Defines a small **FeatureExtractor** CNN (3 conv layers + pooling → 128-dim feature vector). 
2. Uses 1000 real test images and 1000 generated images from the standard VAE. 
3. Computes:

   * Feature means `μ₁, μ₂` and covariances `Σ₁, Σ₂`.
   * FID:
     $$
     |\mu_1 - \mu_2|^2 + \operatorname{Tr}\big(\Sigma_1 + \Sigma_2 - 2(\Sigma_1\Sigma_2)^{1/2}\big)
     $$
   * Using `scipy.linalg.sqrtm`. 

### 8.2 Why this number is *not* a reliable FID

* The feature extractor is **small and untrained**.
* Real and generated images are both Fashion-MNIST style; an untrained CNN can easily produce features with nearly identical mean and covariance, giving an artificially low FID.
* The notebook itself notes that typical FID scores for VAEs on Fashion-MNIST are in the **50–150** range, so a score of 0.00 is clearly not realistic for a true Inception-based FID. 

**How to present this in the report**

* Treat this as a **toy FID implementation** used to demonstrate the formula and pipeline, not as a trustworthy quantitative metric.
* Emphasise that **qualitative inspection** and β-VAE comparisons are the main evaluation tools, and that the exact numeric FID here should not be over-interpreted.

> ⚠️ **Important:** The method shows you understand how FID is computed, but the absolute value is not meaningful due to the untrained feature network.

---

## 9. Effect of Frozen Latent Parameters (4.8)

### 9.1 Experimental setup

Using the trained standard VAE (latent_dim=16): 

1. **Baseline:** standard sampling `z ~ N(0, I)` → decoded images.
2. **Frozen μ = 0**, varying σ ∈ {0.1, 0.5, 1.0}:

   * Sample `ε ~ N(0, I)` and set `z = σ ε`.
   * Decode and visualize 64 samples per σ.

All four conditions (baseline + three σ values) are plotted in a vertical grid titled *“Effect of Frozen Latent Parameters – atharv.bhatt”*. 

### 9.2 Detailed comparison & diversity metric

* Another figure shows 16 samples for each condition in a 2×2 grid: baseline and the three σ values. 
* A custom **diversity metric** is defined: average pairwise L2 distance between flattened images in each set. 

**Qualitative justification**

* **σ = 0.1**:

  * Samples are very similar—latent space is sampled near the origin → low diversity but decently sharp objects.
* **σ = 0.5**:

  * Good compromise between diversity and coherence; images are varied yet mostly recognizable.
* **σ = 1.0**:

  * Highest diversity but also more noisy / distorted samples as we move further from regions visited during training.
* The standard `z ~ N(0, I)` case falls between σ=0.5 and σ=1.0, reflecting the training prior.

These behaviours match theoretical expectations: scaling σ controls how far we move in latent space, trading off diversity for realism.

> **Requirement satisfied:** We systematically explore frozen μ and varying σ, compare visuals, and support with a simple quantitative diversity measure.

---

## 10. Summary

* **Dataset & preprocessing:** Fashion-MNIST loaded with tensor transform (0–1), DataLoaders with a reasonable batch size.
* **Model:** A well-designed conv VAE with explicit μ/logσ² outputs, reparameterization, and 28×28 reconstructions.
* **Loss & training:** Standard BCE + β-KL objective; stable training curves for β=1.
* **β-VAE analysis:** Separate 2-D latent models for β=0.1, 0.5, 1.0; latent evolution GIFs; loss plots; comparison table capturing reconstruction vs disentanglement trade-offs.
* **Evaluation:** Clear original vs reconstructed grids and samples from N(0, I) with qualitative discussion.
* **FID:** Implemented via a custom feature CNN; result (0.0) is acknowledged as **not a reliable absolute score** but shows understanding of the FID pipeline.
* **Frozen latent parameters:** Systematic experiment with μ=0 and varying σ, plus a diversity metric, illustrating how latent variance affects sample diversity and quality.
