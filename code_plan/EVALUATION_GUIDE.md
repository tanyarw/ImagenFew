# Fine-tuning

```nohup python run.py --subset_p 0.1 --model_ckpt ./models_ckpt/ImagenFew/ImagenFew_12.ckpt --config configs/finetune/Rainfall.yaml > rainfall_finetune.log 2>&1 &```

# 📊 Evaluation Methodology Guide: Time Series Generation

This document outlines the evaluation framework for the `ImagenFew` project, with a specific focus on **Sparse Data (e.g., Rainfall)**.

Reliably evaluating generative models for sparse data is challenging because traditional metrics can be easily "fooled" by the abundance of zeros. This guide explains our two-tiered approach: **Standard Global Metrics** and **Conditional Fidelity Metrics**.

---

## 🏗️ Part 1: Standard Global Metrics
These metrics evaluate the model's ability to capture the overall distribution of the dataset.

### 1. Discriminative Score (`test/disc_mean`)
*   **Methodology:** A post-hoc RNN (GRU) classifier is trained to distinguish between real and synthetic sequences. The score is calculated as $|0.5 - \text{Accuracy}|$.
*   **Ideal Value:** **0.0** (meaning the discriminator is guessing at 50% accuracy).
*   **Interpretation:** A low score indicates that the synthetic data is statistically similar to the real data across the entire test set.
*   **⚠️ The Sparse Data Trap:** In rainfall data (which is ~90% zero), a model that generates only zeros can achieve a very good discriminative score because it is correct 90% of the time.

### 2. Predictive Score (`test/pred_mean`)
*   **Methodology:** An RNN is trained on **synthetic** data to predict the next time step. This trained model is then tested on **real** data. The score is the Mean Absolute Error (MAE).
*   **Ideal Value:** Lower is better (typically $< 0.1$ for normalized data).
*   **Interpretation:** Measures how well the synthetic data preserves temporal dependencies for downstream tasks like forecasting.

### 3. Context FID (`test/context_fid`)
*   **Methodology:** We use a pretrained `TS2Vec` encoder to map sequences into a high-dimensional latent space and calculate the Frechet Inception Distance (FID) between the real and synthetic embeddings.
*   **Ideal Value:** **0.0**.
*   **Interpretation:** Captures high-level semantic features and "style" rather than point-by-point values.

---

## 🌧️ Part 2: Conditional Fidelity Metrics (The "Honest" Check)
Designed specifically for sparse data to ensure the model isn't just "cheating" with zeros. Execute via `evaluate_conditional.py`.

### 1. Wet-Window Discriminative Score
*   **Methodology:** The standard Discriminative Score, but calculated **only** on "Wet Windows" (sequences containing at least one rainfall event above a threshold).
*   **Interpretation:** This is the most critical metric for rainfall. If your global score is $0.0045$ but your Wet-Window score is $0.25$, your model has learned "silence" but not "storms."

### 2. Intensity Distribution JSD
*   **Methodology:** We extract all non-zero values from both datasets and calculate the **Jensen-Shannon Divergence** between their histograms.
*   **Interpretation:** Measures if the "heaviness" of the rain is correct. A high JSD means the model might be generating drizzle when it should be generating heavy rain.

### 3. Extreme Value Capture (99th Percentile Ratio)
*   **Methodology:** The ratio of the 99th percentile of synthetic data to the 99th percentile of real data.
*   **Ideal Value:** **1.0**.
*   **Interpretation:** Tests if the model can generate the "tails" of the distribution. If this ratio is $0.1$, the model never generates heavy extremes.

### 4. Event Duration Statistics
*   **Methodology:** We calculate the average length of consecutive non-zero time steps (storms).
*   **Interpretation:** Tests temporal persistence. If real storms last 6 hours but synthetic ones last 1 hour, the model is failing to capture the physical reality of weather systems.

---

## 📈 Interpretation Summary Table

| Metric | "Good" Result | Indicates... |
| :--- | :--- | :--- |
| **Global Disc Score** | $< 0.1$ | Overall statistical matching. |
| **Wet-Window Disc** | $< 0.1$ | Realism during actual events. |
| **Intensity JSD** | $< 0.05$ | Correct rainfall amounts/magnitudes. |
| **Extreme Ratio** | $0.8 - 1.2$ | Successfully captures heavy storms. |
| **Duration Match** | $\pm 10\%$ | Correct temporal persistence. |

---

## 🚀 How to Run the Pipelines

### Standard Evaluation & Visualization
```bash
python visualize.py --model_ckpt <ckpt> --config <config>
```

### Conditional (Sparse Data) Evaluation
```bash
python evaluate_conditional.py --model_ckpt <ckpt> --config <config>
```
