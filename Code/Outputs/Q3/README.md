# Time Series Forecasting of GitHub Stars  

This README documents the full approach and outputs for **Time Series Forecasting – Cumulative GitHub Stars**.  

---

## 1. Problem Overview

We are given cumulative GitHub star time series:
$$
y_t^{(i)} = \text{total stars for repo } i \text{ up to time } t
$$
for multiple repositories.

For **Part 2** we are required to:

- Pick **two repositories** and forecast their future star trajectories.
- Compare **classical ARMA/ARIMA** against **deep learning** models (RNN and 1D CNN).
- Evaluate both **single-step** and **multi-step** forecasts using standard error metrics.

In addition, we must:

- Explain data cleaning and feature engineering,
- Describe model architectures and hyperparameters,
- Provide quantitative metrics and forecast plots,
- Ensure everything is **reproducible**.

---

## 2. Data and Preprocessing

### 2.1 Datasets

- `Data/Q3/stars_data.csv`  

  Columns:
  - `timestamp` – daily timestamps.
  - `stars` – cumulative star count at that timestamp.
  - `repository_id` – GitHub repo identifier.

- `Data/Q3/repo_metadata.json` (optional)  
  Used only to inspect languages/topics; not used directly as features.

### 2.2 Repository Selection

We restrict our analysis to:

1. `facebook/react`
2. `pallets/flask`

All filtering is done inside the notebook using:

```python
df[df["repository_id"] == repo_id]
````

### 2.3 Basic Cleaning

1. **Drop exact duplicates**

   We first remove any duplicate `(timestamp, repository_id)` rows:

   ```python
   df = df.drop_duplicates(subset=["timestamp", "repository_id"])
   ```

2. **Inspect capped values at 4000**

   We compute `max(stars)` per repo. Several repos are capped at `4000`, with long flat tails (e.g., the tensorflow example in the description).
   We note these repos but **only work with React and Flask**, so we do not explicitly remove other repos.

   For React/Flask, the plateau at the end represents a realistic saturation of growth. It does not harm the model:

   * Increments during the plateau are `0`, which is correct.
   * Including them simply teaches the model that the series can level off.

3. **Daily alignment & missing values**

   For each selected repo we create a **daily cumulative series**:

   ```python
   s = repo_df.set_index("timestamp")["stars"].sort_index()
   s = s.asfreq("D")    # reindex to daily frequency
   s = s.ffill()        # forward-fill missing days
   ```

   This:

   * Aligns all time steps on a common daily grid,
   * Handles missing days by assuming “no new stars → cumulative stays constant”.

### 2.4 Cumulative vs Increment Domain

We then derive **daily increments**:

$$
\Delta y_t = y_t - y_{t-1}
$$

in code:

```python
incr = cum_series.diff().fillna(0.0)
```

**Reasoning:**

* Raw cumulative series are strongly trending and non-stationary.
* Increments are closer to stationary and easier for ARIMA and neural models to fit.
* However, evaluation is more intuitive in cumulative space, so we:

  * **Train models on increments**,
  * **Convert predicted increments back to cumulative** for metrics and plots.

### 2.5 Train / Validation / Test Splits

For each repo we split the **cumulative series** chronologically:

* Train: first 70%
* Validation: next 15%
* Test: last 15%

This respects temporal order and mimics real forecasting:

```python
n = len(cum)
idx_train_end = int(0.7 * n)
idx_val_end   = int(0.85 * n)

cum_train = cum.iloc[:idx_train_end]
cum_val   = cum.iloc[idx_train_end:idx_val_end]
cum_test  = cum.iloc[idx_val_end:]
```

We then compute increments **within each segment**:

```python
incr_train = cum_train.diff().fillna(0.0)
incr_val   = cum_val.diff().fillna(0.0)
incr_test  = cum_test.diff().fillna(0.0)
```

This avoids any differencing that crosses split boundaries and prevents leakage from future to past.

### 2.6 Scaling / Normalization

We standardize increments using **only the training segment**:

```python
mu = incr_train.mean()
sigma = incr_train.std()
if sigma == 0:
    sigma = 1.0

incr_train_norm = (incr_train - mu) / sigma
incr_val_norm   = (incr_val   - mu) / sigma
incr_test_norm  = (incr_test  - mu) / sigma
```

Justification:

* For gradient-based DL models, having features with mean ~0 and std ~1 stabilizes training.
* Using training statistics only ensures **no information from validation/test leaks** into model training.

We store `(mu, sigma)` per repo and always **invert** normalization before computing cumulative predictions and metrics:

```python
pred_incr = pred_norm * sigma + mu
```

### 2.7 Sliding-Window Dataset for DL Models

For DL models we work on normalized increments and create **supervised windows**:

* Input (history): last `L` increments,
* Target: next increment.

We use `L = 30` (one month of history) for both RNN and CNN.

Construction:

```python
def _make_windows(arr, L):
    X, y = [], []
    for i in range(L, len(arr)):
        X.append(arr[i-L:i])
        y.append(arr[i])
    return np.stack(X), np.array(y, dtype=np.float32)
```

### 2.8 Visualization & Time-Domain Discussion

For each repo we create plots of:

1. **Cumulative stars vs time**, with vertical lines marking train/val/test splits.
2. **Daily increments vs time**, again with split boundaries.

We also add the required username annotation to every plot:

```python
def add_username(tag="atharv.bhatt"):
    plt.text(
        0.95, 0.95, tag,
        ha="right", va="top",
        transform=plt.gca().transAxes,
        fontsize=10, color="gray", alpha=0.7
    )
```

From these plots we observe:

* React and Flask both show rapid growth followed by slower growth and eventual plateau.
* Splitting near the end means:

  * Training sees both growth and some saturation,
  * Test performance measures how well models extrapolate near the plateau, which is realistic for forecasting GitHub popularity.

---

## 3. Models and Architectures

We implement three models per repository:

1. **ARIMA** on increments (classical baseline),
2. **RNN forecaster** (GRU),
3. **1D CNN forecaster**.

### 3.1 ARIMA (Classical Model)

We fit ARIMA on **train+val increments** (raw, not normalized).

Candidate orders:

* $(p, 0, q) ∈ { (1,0), (2,0), (2,1) }$

For each candidate we:

1. Fit `ARIMA(incr_train_val, order=(p,0,q))`,
2. Forecast `len(val)` steps,
3. Compute RMSE on validation increments,
4. Select order with lowest validation RMSE.

Chosen models (example from my runs):

* React: ARIMA(1,0,0)
* Flask: ARIMA(2,0,1)

Number of parameters of these ARIMA models is small (a few autoregressive + moving-average coefficients plus intercept), making them **very parsimonious** baselines.

### 3.2 RNN Forecaster (Deep Learning)

Architecture:

* **Input:** window of length `L = 30` normalized increments.
* Reshape: `(batch, L)` → `(batch, L, 1)`.
* **GRU layer:**

  * `input_size = 1`
  * `hidden_size = 32`
  * `num_layers = 1`
* Take the last hidden state and pass through:
* **Fully connected layer:** `hidden_size → 1`
* **Output:** scalar prediction of next normalized increment.

Code (core):

```python
class RNNForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_seq = x.unsqueeze(-1)           # (batch, L, 1)
        out, _ = self.rnn(x_seq)         # (batch, L, hidden_size)
        last_h = out[:, -1, :]           # final hidden state
        return self.fc(last_h).squeeze(-1)
```

* **Parameters:** ~3,400 trainable parameters.
* **Loss:** MSE on normalized increments.
* **Optimizer:** Adam, learning rate `1e-3`.
* **Epochs:** 50, with best checkpoint chosen via validation MSE.

Justification:

* GRU can capture nonlinear temporal dependencies in star increments (e.g., bursts, slowing growth).
* Small hidden size keeps the model light and helps avoid overfitting.

### 3.3 1D CNN Forecaster (Deep Learning)

Architecture:

* **Input:** `(batch, L)` normalized increment sequence.
* Reshape to `(batch, 1, L)` and apply:

  1. `Conv1d(1, 16, kernel_size=3, padding=1)` + ReLU
  2. `Conv1d(16, 32, kernel_size=3, padding=1)` + ReLU
  3. **Global average pooling** over time dimension → `(batch, 32)`
  4. `Linear(32, 1)` → scalar normalized increment.

Code (core):

```python
class CNN1DForecaster(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)        # (batch, 1, L)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=2)         # global average pooling
        return self.fc(x).squeeze(-1)
```

* **Parameters:** ~1,700 trainable parameters.
* Training setup is identical to the RNN (MSE + Adam + 50 epochs).

Justification:

* CNNs capture **local temporal patterns** (e.g., 3–7 day bursts) via convolution.
* Global average pooling enforces a compact summary, improving parsimony.

### 3.4 Loss Function Choice

We use **MSE loss** on increments for DL models:

$$
\mathcal{L} = \frac{1}{N} \sum_{t} (\Delta \hat{y}_t - \Delta y_t)^2
$$

* Squaring errors penalizes large mistakes more strongly.
* Works nicely with normalized inputs and the Adam optimizer.
* For evaluation we convert back to cumulative stars and report **MAE** and **RMSE**, which are more interpretable in the original scale.

---

## 4. Evaluation Protocol

We evaluate models in two regimes:

1. **Single-step prediction across the entire test horizon**
2. **Multi-step autoregressive forecasting with increasing horizon**

All metrics are computed in **cumulative star space**.

### 4.1 Reconstructing Cumulative from Increments

Given predicted increments $Δŷ_t$ and the last true cumulative value before the forecast region $y_T$, we reconstruct:

```python
cum_pred = np.cumsum(pred_incr) + y_T
```

This is used for **all models**, ensuring a fair comparison.

### 4.2 Single-Step Evaluation

For each repo:

* **ARIMA:**

  * Fit once on train+val increments.
  * Forecast `len(test)` daily increments.
  * Reconstruct cumulative series starting from last train+val cumulative.
* **RNN/CNN:**

  * Build sliding windows on **test normalized increments**, length `L=30`.
  * Predict normalized increments, denormalize using `(μ, σ)`.
  * Reconstruct cumulative starting from last train+val cumulative.

We then align and compute:

* Mean Absolute Error (MAE) on cumulative,
* Root Mean Squared Error (RMSE) on cumulative.

Example (qualitative):

* For **React**:

  * ARIMA has noticeably larger MAE/RMSE, reflecting difficulty in capturing smooth nonlinear growth.
  * RNN and CNN have very small errors; CNN is slightly worse than RNN for React.
* For **Flask**:

  * ARIMA again underperforms.
  * Both DL models fit the curve very closely; CNN slightly outperforms RNN with fewer parameters.

We also produce **test forecast plots**:

* x-axis: date,
* y-axis: cumulative stars,
* lines: true vs ARIMA vs RNN vs CNN,
* vertical line at test start,
* username annotation `atharv.bhatt`.

### 4.3 Multi-Step Autoregressive Forecasting

To analyze error growth with horizon ( h ), for each repo we:

1. Use only **train+val** increments to form the initial history.

   * For DL models: take the last `L=30` normalized increments as the starting window.
   * For ARIMA: no window; we just call `forecast(h)` from the fitted model.

2. For horizons $h \in {1, 3, 7, 14, 30}$:

   * **ARIMA:**

     * `forecast(h)` → `h` increments → reconstruct cumulative.
   * **RNN/CNN:**

     * Autoregressively roll out:

       * At each step, feed the current window,
       * Append predicted normalized increment to the history,
       * Denormalize increments to reconstruct cumulative.

3. Compare the predicted cumulative sequence of length `h` with the **first `h` days of the test cumulative**.

4. Compute MAE for each horizon and model.

We then plot **error vs horizon**:

* x-axis: horizon (days ahead),
* y-axis: MAE in cumulative stars,
* curves: ARIMA, RNN, CNN.

Qualitative observations:

* For both repos, **ARIMA error grows faster** with horizon, especially at larger `h`.
* **RNN and CNN remain stable** for short horizons (1–7 days) and grow more gracefully with horizon.
* Between RNN and CNN:

  * On Flask, CNN is slightly more accurate at longer horizons with fewer parameters, showing better parsimony.
  * On React, RNN slightly edges out CNN in some horizons, but both far outperform ARIMA.

Note: For simplicity, we use **one forecast origin** (the first day of the test period). A more complex backtesting procedure would re-forecast from multiple origins and average errors, but the current setup still captures the trend of error growth with horizon.

---

## 5. Results and Discussion

### 5.1 Quantitative Summary (Single-Step)

For each repo and model we get:

* **React** (qualitative shape):

  * ARIMA: significantly larger MAE/RMSE.
  * RNN: smallest errors; nearly tracks the cumulative curve.
  * CNN: close to RNN but slightly worse.

* **Flask**:

  * ARIMA: larger MAE/RMSE than DL models.
  * RNN: very small MAE/RMSE.
  * CNN: even slightly better MAE/RMSE than RNN with fewer parameters.

This supports the intuition that:

* The star trajectories have **nonlinear, saturating growth** that simple ARMA struggles to fit.
* Small neural models (RNN/CNN) can learn these patterns effectively with relatively few parameters.

### 5.2 Error vs Horizon (Multi-Step)

From the error-v-horizon plots:

* All models’ errors increase with horizon (as expected).
* ARIMA’s errors grow steeply, indicating poorer long-term extrapolation.
* RNN and CNN show **slower error growth**, so they generalize better as we forecast further into the future.
* CNN’s lower parameter count but comparable or slightly better performance (especially for Flask) highlights a favorable **complexity–accuracy trade-off**.

### 5.3 Parsimony and Stability

* **Parsimony:**

  * ARIMA has very few parameters but poor fit; it is parsimonious but underfits the nonlinear trend.
  * RNN (~3.4k params) and CNN (~1.7k params) are still very small models.
  * On Flask, CNN gives the best performance with roughly half the parameters of the RNN → **best parsimony** among DL models.
* **Stability:**

  * The error-vs-horizon curves for RNN/CNN are smooth and gradually increasing → stable forecasts.
  * ARIMA error is higher and increases more erratically with horizon.

From this we can say:

> The dataset exhibits **smooth, saturating temporal dynamics** that are better captured by small nonlinear models than by purely linear ARMA. An expressive but compact CNN (or RNN) offers the best balance between model complexity and predictive accuracy.

---

## 6. Reproducibility

To ensure the results are fully reproducible:

1. **Random Seeds**

   In `3.ipynb` we fix seeds at the top:

   ```python
   SEED = 42
   np.random.seed(SEED)
   random.seed(SEED)
   torch.manual_seed(SEED)
   ```

   This keeps splits, initialization, and training behaviour consistent across runs.

2. **Deterministic Splits & Normalization**

   * Train/val/test splits are deterministic (based on fixed fractions and ordering).
   * Normalization parameters `(μ, σ)` are computed **once from training data** and stored.
   * Both ARIMA and DL models reuse the same splits and sequences.

3. **Single Notebook**

   The entire pipeline is implemented in **one Jupyter notebook**:

   * data loading,
   * preprocessing,
   * model definitions,
   * training loops,
   * evaluation & plotting.

   Running `3.ipynb` from top to bottom reproduces all metrics and plots.

4. **Plot Annotation**

   Every plot calls `add_username("atharv.bhatt")` to place my username in the top-right corner, as required.

---

## 7. Conclusion

This solution:

* Cleans and aligns GitHub star time series for `facebook/react` and `pallets/flask`,
* Uses increments + normalization (based on training data) to create a stable modeling domain,
* Implements and compares **ARIMA**, **RNN**, and **1D CNN** forecasters,
* Evaluates them with **MAE/RMSE** on cumulative stars for both **single-step** and **multi-step** forecasting,
* Demonstrates that small deep models significantly outperform a classical ARIMA baseline while remaining parameter-efficient,
* Provides all code and details needed for full reproducibility.
