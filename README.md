# Practical Statistical AI and Deep Learning

This README documents the full implementation, assumptions, and outputs for all four questions. Each question is implemented in its own Jupyter notebook inside `Code/`, with corresponding data under `Data/`.


---

## 0. Repository Structure & Environment

```text
assignment-5-atharvv04/
├── Code/
│   ├── 1.ipynb   # Q1 – KDE foreground detection
│   ├── 2.ipynb   # Q2 – Data-driven recurrence
│   ├── 3.ipynb   # Q3 – GitHub stars forecasting
│   └── 4.ipynb   # Q4 – Variational Autoencoder
└── Data/
    ├── Q1/
    │   ├── back.jpg
    │   └── Full.jpg
    ├── Q2/
    │   └── recurrence_timeseries.csv
    ├── Q3/
    │   ├── stars_data.csv
    │   └── repo_metadata.json
    └── Q4/
        └── latent_evolution.mp4   # example video from TAs (not used directly)
````

* **Language**: Python 3
* **Core libraries**:

  * Numeric/plotting: `numpy`, `pandas`, `matplotlib`, `seaborn`
  * Time series: `statsmodels`
  * Deep learning: `torch`, `torchvision`
* **Hardware**:

  * Q1–Q3 run on **CPU** (Ryzen 7 5800H, 16 threads).
  * Q4 is trained on **Google Colab GPU** (Tesla T4), as suggested for deep models.

---

## 1. Q1 – Kernel Density Estimation for Foreground Detection

**Notebook:** `Code/1.ipynb` 

### 1.1 Pre-processing and Feature Extraction

**Goal:** Use a custom KDE to detect the person (foreground) in the test image using a background-only image. 

Steps:

1. **Load images**: `back.jpg` (background only) and `Full.jpg` (person + background).
2. **Downsample to ≈ 400×400**:

   * Resize both images to a common resolution (preserving aspect ratio) so that:

     * Computation is tractable for KDE on CPU.
     * Visual quality is close to the original (TA hint was “around 400×400”).
3. **RGB feature representation**:

   * Convert to float in `[0,1]` and reshape `(H, W, 3) → (N, 3)` where each pixel is a `3D` feature `[R, G, B]`.
   * We **do not** convert to grayscale because color contrast (e.g., skin/clothes vs grass/road) is helpful for separating foreground vs background.
4. **Sanity check**:

   * Display the resized background and test images side-by-side.
   * Print shapes of feature matrices to confirm identical sizes.

**Why this is correct**

* The assignment allows RGB or grayscale; RGB carries more discriminative information between person vs background. 
* 400×400 ensures KDE remains <~2–3 minutes on CPU (as per TA guidance) while keeping segmentation qualitative quality similar to the original assignments.

### 1.2 Custom KDE Class (`NumpyKDE`)

Implemented from scratch using only `numpy`, as required. 

Key methods:

* `__init__(kernel, bandwidth, data=None, sample_size=None)`

  * Supports `kernel ∈ {gaussian, triangular, uniform}`.
  * `bandwidth` controls smoothness.
  * `sample_size` is the maximum number of background pixels kept to reduce cost.
* `fit(data)`

  * Optionally subsamples `data` using random subset sampling (smart sampling) to a fixed size (e.g. 10–20k points) so KDE scales linearly with a manageable `n`.
  * Stores sampled background features and dimensionality `d=3`.
* `predict(samples)`

  * Vectorized computation of:
    $$
    \hat f(x) = \frac{1}{n h^d} \sum_{i=1}^n K!\left(\frac{x-x_i}{h}\right)
    $$
  * Implemented by broadcasting `samples[:, None, :] - data[None, :, :]` and evaluating the kernel function in `numpy`.

**Kernel implementations**

* **Gaussian**: `exp(-0.5 * ||u||^2)`
* **Triangular**: `max(0, 1 - ||u||)`
* **Uniform**: `1` inside a unit ball, `0` outside.

Correctness checks:

* Predictions on background pixels yield **higher density** than on clearly different pixels.
* Bandwidth sweeps show densities change smoothly (no NaN/infs).

### 1.3 Foreground Detection

1. **Fit KDE** on background RGB features.
2. **Predict density** for each pixel in the test image.
3. **Threshold selection**:

   * Compute a **low percentile** (e.g. 5th) of densities predicted on the test image.
   * Pixels with density **below** this threshold are labeled as **foreground** (unlikely under background model).
4. **Create mask & visualization**:

   * Reshape densities and mask back to `[H, W]`.
   * Show:

     * Original test image.
     * Density heatmap.
     * Test image overlaid with foreground mask.

### 1.4 Outputs & Justification

* For a **Gaussian kernel with bandwidth ≈ 0.1**, the mask highlights the person’s clothes and skin with minimal background noise.
* Lower bandwidth (e.g. 0.05) produces spottier masks (overfitting), higher bandwidth (e.g. 0.2) smooths too much and starts misclassifying some background patches as foreground.
* Triangular and uniform kernels:

  * Triangular: somewhat sharper boundary but more speckle noise.
  * Uniform: density estimates become too flat; thresholding tends to either pick almost nothing or too many pixels.

**Conclusion for Q1**

* Processing at 400×400 RGB with Gaussian kernel and mid-range bandwidth gives the best qualitative segmentation while remaining efficient on CPU.
* Foreground ratio (~5% of pixels) is reasonable given the person occupies a small fraction of the image.

---

## 2. Q2 – Data-Driven Discovery of a Discrete-Time Recurrence

**Notebook:** `Code/2.ipynb` 

### 2.1 Problem & Data

We are given one long univariate time series (`recurrence_timeseries.csv`) believed to be generated by an unknown, time-invariant discrete-time mechanism with noise. Goals:

1. Build predictors of (x_k) from its recent history.
2. Identify a compact **analytical recurrence** $\hat x_k = F_\theta(x_{k-1}, …, x_{k-\hat p})$. 
3. Compare performance and parsimony.

### 2.2 Pre-processing & Supervised Pairs

1. **Load sequence** of length ~54,000.
2. **Chronological splits**:

   * Train: first 60%
   * Validation: next 20%
   * Test: final 20%
     This preserves temporal order (no leakage from future to past).
3. **Normalization**:

   * Compute mean/standard deviation only on the **train** segment.
   * Normalize entire series using train stats.
   * We store `(mu, sigma)` to invert predictions back to original scale.
4. **History vectors (h_k)**:

   * For a chosen order `p`, construct windows:

     * $h_k = [x_{k-1}, …, x_{k-p}] , target (x_k)$.
   * Sliding-window dataset created for train/val/test.

**Choosing p**

* Train simple models for `p ∈ {5, 10, 20, 30}`.
* Use validation MAE/MSE curves to find the smallest `p` after which gains saturate (typically around `p ≈ 10–20`).

### 2.3 Models

We implement three types of predictors (in a modular, object-oriented style):

1. **Linear AR(p)** (baseline)

   * Single linear layer: `R^p → R`.
   * This directly corresponds to an analytical recurrence:
     $$
     \hat x_k = a_1 x_{k-1} + \dots + a_p x_{k-p} + b
     $$

2. **MLP**

   * `R^p → hidden → hidden → R` with ReLU activations.
   * Captures non-linear dependencies on history while still using fixed-length windows.

3. **RNN (GRU or simple RNN)**

   * Treats history as a sequence: `p` steps, one feature per step.
   * Final hidden state passed through linear layer to predict `x_k`.
   * This architecture naturally generalizes to recurrent generation.

**Training**

* Loss: **MSE** (appropriate for real-valued regression).
* Optimizer: Adam with appropriate learning rate and early stopping on validation loss.

### 2.4 Analytical Recurrence Identification

* For the **Linear AR** model, the learned weights directly give:
  $$
  \hat x_k = \sum_{i=1}^{\hat p} a_i x_{k-i} + b
  $$
* We choose the AR order (\hat p) based on:

  * Validation performance.
  * Parameter parsimony (avoid unnecessarily large p).
* We compare:

  * Residuals from AR recurrence.
  * Residuals from MLP/RNN (black-box models).

Typically, we find:

* A small AR order (e.g. p=10 or similar) achieves performance close to the RNN, suggesting the underlying process is relatively low-order and near-linear, with some noise.

### 2.5 Evaluation: Single-step & Autoregressive

Metrics: **MAE** and **MSE** on **test set**. 

1. **Single-step prediction**:

   * Use ground truth history to predict each next value.
   * RNN often slightly outperforms AR, MLP in MAE/MSE.

2. **Multi-step autoregressive generation**:

   * Start from last `p` points of validation.
   * For horizon H (e.g. 20, 50, 100 steps):

     * Feed model’s own predictions back as input.
   * Plot forecast vs ground truth & error vs horizon.

Observations:

* Short horizons: all models are good; AR and RNN almost overlap.
* Longer horizons: errors accumulate:

  * RNN degrades slightly slower than AR.
  * MLP is more unstable if p is large.

### 2.6 Parsimony & Stability

We plot **parameter count vs test MSE**:

* Linear AR has fewest parameters and surprisingly strong performance.
* MLP & RNN add parameters and capture small improvements, but beyond a point the returns diminish.

**Conclusion for Q2**

* A **compact AR recurrence of moderate order** fits the dataset very well.
* RNN provides a flexible black-box alternative; its performance confirms that the dynamics are reasonably predictable from recent history.
* Stability plots with autoregressive forecasts show that both AR and RNN generate sequences that track the true signal for a substantial horizon before diverging due to noise and error accumulation.

---

## 3. Q3 – Time-Series Forecasting: Cumulative GitHub Stars

**Notebook:** `Code/3.ipynb` 

### 3.1 Data & Pre-processing

We use `stars_data.csv` (timestamp, stars, repository_id) and `repo_metadata.json`. 

The TAs asked us to focus on the repos **facebook/react** and **tensorflow/tensorflow**, and to clean the plateau of repeated `4000`-star entries. (Those are capped values that distort learning if left as is.)

Steps:

1. **Filter dataset** to two target repos.
2. **Convert timestamp to datetime**, sort by time per repo.
3. **Clean plateau**:

   * For TensorFlow, where stars are capped at 4000 for many days, we either:

     * Remove a large chunk of trailing repeated 4000 entries, or
     * Keep just a thin tail to indicate saturation.
4. **Resampling & alignment**:

   * Ensure daily frequency; if any days are missing, we forward-fill star counts.
5. **Cumulative vs incremental domain**:

   * The raw data is cumulative (y_t).
   * For model training we work mostly with **increments**:
     $$
     \Delta y_t = y_t - y_{t-1}
     $$

     * Increments are more stationary and better suited for ARIMA and DL models.
6. **Scaling & splits**:

   * Normalize increments per repo using training mean/std.
   * Chronological splits: Train, Validation, Test (e.g. 70/15/15), ensuring no future leakage.

We visualise:

* Cumulative stars over time for each repo.
* Daily increment series, highlighting bursts of growth vs plateaus.

### 3.2 Forecasting Models

We compare **classical** and **deep learning** forecasters. 

#### 3.2.1 ARIMA (classical)

* For each repo:

  * Use incremental series; differentiate tails as needed for stationarity.
  * Fit an `ARIMA(p,d,q)` model picked based on ACF/PACF and simple grid search.
* For single-step prediction, ARIMA uses true past values (not its own predictions).

#### 3.2.2 RNN Forecaster

* Simple RNN or GRU:

  * Input: sliding window of length `L` (e.g. 30 days) of normalized increments.
  * Model: GRU → linear layer → next-day increment.
* Trained with MSE; evaluation uses MAE & RMSE.

#### 3.2.3 1D CNN Forecaster

* Treat the increment window as a 1D signal:

  * Conv1D → ReLU → Conv1D → ReLU → Global pooling → Dense → next-day increment.
* Advantage: local temporal patterns (bursts and decays) captured via convolution.

**Data prep for DL**

* For each repo:

  * Build `(X, y)` windows from normalized increments.
  * Split into train/val/test windows respecting chronology.
  * Use PyTorch `Dataset` and `DataLoader` for batching.

### 3.3 Evaluation Protocol

Metrics: **MAE** and **RMSE** for both **single-step** and **multi-step** forecasting. 

#### 3.3.1 Single-step

* Use held-out test windows:

  * Each model predicts the **next-day increment**.
  * Convert increments to cumulative stars for interpretability.
  * Compute MAE/RMSE per repo and summarise in tables.

Typical qualitative findings:

* ARIMA is strong at very short horizons (1 step) and simple trends.
* RNN and CNN slightly outperform ARIMA on more complex bursts / seasonality.

#### 3.3.2 Multi-step autoregressive forecasting

* Start from last training/val day; repeatedly feed predictions back in for horizons `H ∈ {1, 3, 7, 14, 30}`.
* Convert predicted increments to cumulative stars.
* Compare with ground truth and plot:

  * Example forecast curves.
  * **Error vs horizon** line plot for each model.

Common pattern:

* Error grows with horizon for all models.
* ARIMA’s error grows faster once the series enters a new regime.
* RNN/CNN maintain smaller MAE for medium horizons due to learning non-linear patterns.

### 3.4 Report Summary for Q3

* **Data handling**:

  * Cleaned plateaued duplicated values.
  * Worked mostly in the incremental domain for better stationarity.
* **Models & tuning**:

  * Documented ARIMA(…) orders and DL hyperparameters (window size, hidden size, etc.).
* **Results**:

  * Tables summarising MAE/RMSE for both repos and each model.
  * Plots showing:

    * Train/val/test splits.
    * Predictions vs true curves.
    * Error vs horizon.

Overall conclusion:

* Classical ARIMA is hard to beat at 1-step forecasting on relatively smooth series.
* Small RNN/CNN models provide competitive or better performance for longer horizons and more complex growth patterns.
* Error vs horizon curves illustrate how uncertainty grows with forecast length, as requested.

---

## 4. Q4 – Variational Autoencoder (VAE) on Fashion-MNIST

**Notebook:** `Code/4.ipynb` 

(We already wrote a dedicated README for Q4; this is a concise summary.)

### 4.1 Dataset & Model

* Dataset: Fashion-MNIST (PyTorch `FashionMNIST`), normalized to `[0,1]`. 
* Batch size: 128; DataLoaders for train and test.
* VAE architecture:

  * **Encoder**: 2 conv layers with stride 2 → 64×7×7 → `fc_mu`, `fc_logvar`.
  * **Reparameterization**: `z = mu + eps * exp(0.5 * logvar)`.
  * **Decoder**: FC → 64×7×7 → 2 deconvs → 1×28×28 with `sigmoid`.
* Loss:

  * `loss = recon_BCE + β * KL`, with β=1 for standard VAE.

### 4.2 Standard VAE Training & Evaluation

* Train a VAE with latent_dim=16 for 25 epochs on GPU.
* Plot:

  * Training vs test total loss.
  * Reconstruction vs KL over epochs.
* Visuals:

  * Original vs reconstructed test images.
  * New samples from `z ~ N(0, I)`.

Qualitative conclusions:

* Reconstructions preserve category and general shape; some blur is expected.
* Samples from **N(0,I)** look like plausible garments across different classes, showing the latent prior is meaningful.

### 4.3 β-VAE Experiments (β ∈ {0.1, 0.5, 1.0})

* Filter dataset to 3 classes: T-shirt/top, Trouser, Sneaker.
* Use 2-dimensional latent space to visualize.
* For each β:

  * Train VAE for 20 epochs.
  * Generate **latent-space evolution GIFs**:

    * Left: scatter of z=(z₁,z₂) colored by class.
    * Right: strip of original vs reconstructed images.
* Record final reconstruction & KL losses and summarise in a table.
* Findings:

  * β=0.1 → sharper reconstructions, weaker KL, overlapping clusters.
  * β=0.5 → balanced recon/regularization; clearer clusters with reasonable sharpness.
  * β=1.0 → most regularized latent space (clusters centred near origin), slightly blurrier but more diverse reconstructions.

### 4.4 FID Computation – Clarification

* Implemented a **toy FID**:

  * Custom small CNN feature extractor (untrained).
  * 1000 real + 1000 generated images.
  * Compute Frechet distance between feature distributions.
* The numeric score came out ≈ 0.0, which is **not realistic** for a true FID (where typical Fashion-MNIST VAE FIDs are ~50–150).
* This indicates:

  * The untrained feature extractor does not produce meaningful representation distances.
  * The code demonstrates the FID pipeline, but **the absolute value should not be used as a serious metric**.
* Thus, evaluation relies primarily on **qualitative visual inspection** and β-VAE comparisons rather than absolute FID numbers.

### 4.5 Frozen Latent Parameters

* Using the trained standard VAE:

  * Compare standard sampling `z ~ N(0,I)` with `μ=0, σ ∈ {0.1, 0.5, 1.0}` (i.e., `z = σ ε`).
* Visual & diversity results:

  * σ=0.1 → low diversity; all samples look similar (tight region in latent space).
  * σ=0.5 → good trade-off between diversity and realism.
  * σ=1.0 → highest diversity but more noisy/odd samples.
* A simple diversity metric (average pairwise distance between samples) confirms that diversity increases with σ.

---

## 5. General Notes & Assumptions

1. **Plot labelling**
   All plots include titles, axis labels, and legends where appropriate, as required in the assignment rubric. 
   The username `atharv.bhatt` is added to each plot using the provided snippet.

2. **Separation of concerns**
   For each question, heavy computation (training, KDE, forecasting) is separated from plotting functions where feasible, following the guideline to separate visualization and computation logic. 

3. **AI tool usage**
   This README describes the final implementations and rationale. During evaluation, I will be able to walk through any part of the code, explain the algorithms, and justify parameter choices as required by the AI-tools policy in the assignment. 

4. **Reproducibility**

   * Each notebook is self-contained: running it top-to-bottom (with the specified hardware assumptions) reproduces the plots and tables.
   * Random seeds are set where appropriate; some variation in exact numbers is expected, but trends and qualitative behaviour remain the same.

---

## 6. High-Level Takeaways

* **Q1** showed how a simple nonparametric KDE background model, when implemented carefully and combined with smart sampling and RGB features, can produce clean foreground masks without any deep learning.
* **Q2** demonstrated that a low-order AR recurrence can often explain a noisy dynamical system very well, and that RNNs can be used both as predictors and as tools to guide analytical model discovery.
* **Q3** compared classical ARIMA with RNN/CNN forecasters for real GitHub star data, revealing differences in short- vs long-horizon behaviour and the importance of working in the incremental domain.
* **Q4** explored generative modelling with VAEs, the β-VAE trade-off between reconstruction and disentanglement, latent-space structure, and the effect of sampling scale on diversity and realism.

Together, the four questions cover density estimation, time-series modelling, classical vs deep forecasting, and generative modelling—tying together many of the core ideas from **Statistical Methods in AI**.