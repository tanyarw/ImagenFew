# Regime-Conditional Synthetic Rainfall Generation: Code & Implementation Guide

This document is a practical developer guide and code walkthrough for the regime-conditional time-series generation pipeline in **ImagenFew**. It covers execution commands, architecture mapping, and step-by-step code implementations across data labeling, dataset construction, shape-safe fine-tuning, autoregressive sampling, and sparsity thresholding.

---

## 1. Quickstart Execution Commands

### Step 1: Label 10-Minute Historical Data
Merge regime labels from the 5-minute training dataset into the 10-minute resolution dataset:
```bash
python scripts/label_10min_data.py
```

### Step 2: Fine-Tune Regime-Conditional Model
Fine-tune the unconditional checkpoint using regime labels ($n\_classes=4$):
```bash
python regime_training/train_regime.py \
    --model_ckpt ./logs/ImagenFew/Rainfall/<run_id>/best_model.pt \
    --config regime_training/config.yaml \
    --epochs 1001 --batch_size 2048 --learning_rate 1e-4
```

### Step 3: Generate Synthetic Rainfall Time Series
Generate multi-year synthetic sequences using the regime-trained checkpoint and saved scaler:
```bash
# Mixed Mode (Proportional regime sampling across a 10-year simulation)
python regime_training/generate_regime.py \
    --model_ckpt logs/ImagenFew/Rainfall_Regime/<run_id>/best_regime_model.pt \
    --scaler_path logs/ImagenFew/Rainfall_Regime/<run_id>/scaler.pkl \
    --years 10

# Single-Regime Mode (Lock generation to Regime 3 extreme cloudbursts for stress-testing)
python regime_training/generate_regime.py \
    --model_ckpt logs/ImagenFew/Rainfall_Regime/<run_id>/best_regime_model.pt \
    --scaler_path logs/ImagenFew/Rainfall_Regime/<run_id>/scaler.pkl \
    --years 5 --regime 3
```

---

## 2. Codebase Architecture & File Mapping

| Component | File Path | Core Responsibility |
| :--- | :--- | :--- |
| **Sequence Tagging** | `scripts/label_10min_data.py` | Merges block-level GMM labels (`gmm_regime` $\in \{0,1,2,3\}$) onto 10-minute timestamps. |
| **Dataset Engineering** | `regime_training/regime_dataset.py` | Implements `RegimeDataset` for block-constrained sliding window extraction and standardization. |
| **Model Training** | `regime_training/train_regime.py` | Fine-tunes `ImagenFew` with `load_checkpoint_safe()` to re-initialize the classification head. |
| **Inference & Sampling** | `regime_training/generate_regime.py` | Executes one-hot conditioned reverse diffusion, block interleaving, and $0.005\text{ mm}$ thresholding. |
| **Configuration** | `regime_training/config.yaml` | Hyperparameters overriding base classes to `n_classes: 4` and setting delay embeddings. |

---

## 3. Stage 1: Data Labeling Implementation (`scripts/label_10min_data.py`)

Historical 10-minute rainfall data (`rainfall_10min.csv`) is unlabeled. The labeling script performs a relational merge against the pre-clustered 5-minute dataset (`train_dataset.csv`), which contains 14-day macro-windows assigned to 4 Gaussian Mixture Model (GMM) seasonal regimes:
- **Regime 0:** Monsoon / Early Summer Transition
- **Regime 1:** Autumn Decay Transition
- **Regime 2:** Winter / Calm Resting State (Low intensity, high sparsity)
- **Regime 3:** Late Summer / Autumn Convective Peak (Extreme cloudbursts)

---

## 4. Stage 2: Block-Constrained Sliding Windows (`regime_training/regime_dataset.py`)

To prevent cross-regime contamination (where a single training window spans two different GMM seasons), `RegimeDataset` slices sequences strictly **within** individual `block_id` boundaries.

### Code Implementation
```python
class RegimeDataset(Dataset):
    def __init__(self, csv_path, seq_len=24, scale=True, scaler=None):
        self.seq_len = seq_len
        df = pd.read_csv(csv_path)

        # 1. Group contiguous time series by block_id
        blocks = []
        for _, group in df.groupby("block_id", sort=True):
            values = group["avg_rainfall"].values.astype(np.float32).reshape(-1, 1)
            regime = int(group["gmm_regime"].iloc[0])  # Class label in {0, 1, 2, 3}
            blocks.append((values, regime))

        # 2. Global standardization across non-grouped continuous data
        if scale:
            all_values = np.concatenate([b[0] for b in blocks], axis=0)
            self.scaler = scaler if scaler is not None else StandardScaler().fit(all_values)

        # 3. Extract sliding windows strictly inside block boundaries
        self.samples, self.labels = [], []
        for values, regime in blocks:
            scaled = self.scaler.transform(values) if scale else values
            n_windows = len(scaled) - seq_len + 1
            for i in range(n_windows):
                self.samples.append(torch.FloatTensor(scaled[i : i + seq_len]))
                self.labels.append(regime)

        self.labels = torch.LongTensor(self.labels)
        self.n_regimes = 4
```

---

## 5. Stage 3: Shape-Safe Checkpoint Loading (`regime_training/train_regime.py`)

When fine-tuning an unconditional or dataset-level `ImagenFew` checkpoint, the categorical linear embedding layer (`map_label`) changes shape from `[channel_dim, 43]` (43 multi-domain datasets) to `[channel_dim, 4]` (4 GMM regimes).

### Code Implementation: `load_checkpoint_safe`
Directly calling `model.load_state_dict()` raises a shape mismatch RuntimeError. The shape-safe loader filters out the mismatched classification head while preserving all U-Net temporal self-attention and delay-embedding STFT weights:

```python
def load_checkpoint_safe(model, ckpt_path, device):
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state = loaded.get("model", loaded)
    cur_state = model.state_dict()

    # Filter state dict by exact parameter shape matching
    filtered = {}
    for k, v in ckpt_state.items():
        if k in cur_state and v.shape == cur_state[k].shape:
            filtered[k] = v
        else:
            # Automatically skips 'map_label.weight' and 'map_label.bias'
            logging.warning("Skipping %s (ckpt=%s vs model=%s)", k, list(v.shape), list(cur_state[k].shape))

    model.load_state_dict(filtered, strict=False)
```

During training, the skipped `map_label` projection is trained from scratch via AdamW optimizer ($\text{lr}=1\times 10^{-4}$) with Exponential Moving Average (EMA) tracking (`ema_warmup: 100`).

---

## 6. Stage 4: Autoregressive Sampling Mechanics (`regime_training/generate_regime.py`)

Inference drives the reverse diffusion process (`DiffusionProcess.interpolate`) by injecting one-hot regime vectors into the U-Net conditioning layers.

### 1. One-Hot Conditioning Injection
```python
# Create batch of target class indices and one-hot encode
cls = torch.full((batch_size,), target_regime, device=device, dtype=torch.long)
one_hot_cond = nn.functional.one_hot(cls, num_classes=args.n_classes).float()

# Initialize Gaussian noise and mask
x_img = torch.zeros(batch_size, channels, args.img_resolution, args.img_resolution, device=device)
mask = model.ts_to_img(torch.zeros(batch_size, args.seq_len, channels, device=device), pad_val=1)

# Reverse diffusion sampling conditioned on regime
with torch.no_grad():
    sampled_img = process.interpolate(x_img, mask, class_labels=one_hot_cond)
    ts_sequence = model.img_to_ts(sampled_img)[:, :, :channels]
```

### 2. Proportional Allocation (Mixed Mode vs. Single-Regime Mode)
```python
# Load empirical regime proportions from training corpus
with open("logs/.../regime_proportions.pkl", "rb") as f:
    regime_proportions = pickle.load(f)  # e.g., {0: 0.255, 1: 0.370, 2: 0.279, 3: 0.096}

if cli.regime is not None:
    # Single-Regime Mode: Lock 100% of samples to a specified class (e.g., Regime 3)
    regime_alloc = {cli.regime: num_samples}
else:
    # Mixed Mode: Distribute sample counts proportionally to training distribution
    regime_alloc = {r: max(1, int(p * num_samples)) for r, p in regime_proportions.items()}
    # Reconcile rounding remainder to exact target length
    diff = num_samples - sum(regime_alloc.values())
    most_common = max(regime_proportions, key=regime_proportions.get)
    regime_alloc[most_common] += diff
```

---

## 7. Stage 5: Block Interleaving & Hard Sparsity Thresholding

After generating latent sequences across all regimes, post-processing restores multi-year temporal transitions and physical zero-fraction sparsity.

### Code Implementation
```python
# 1. Concatenate generated sequence blocks across regimes -> [N, seq_len, 1]
generated = np.concatenate(all_generated, axis=0)

# 2. Block Interleaving: Randomly permute blocks to simulate natural weather transitions
if cli.regime is None:
    rng = np.random.default_rng(cli.seed)
    generated = generated[rng.permutation(len(generated))]

# 3. Unroll to 1D continuous series and apply inverse standard scaling
continuous = generated.reshape(-1, channels)[:total_steps]
unscaled = scaler.inverse_transform(continuous)

# 4. Hard Sparsity Thresholding: Remove Gaussian background noise haze
unscaled[unscaled < 0.005] = 0.0
```

---

## 8. Configuration Reference (`regime_training/config.yaml`)

Key configuration parameters overriding base training for 4-class GMM regime conditioning:
```yaml
# --- Data & Conditioning ---
train_csv: data/rainfall/train/rainfall_10min_labeled.csv
test_csv: data/rainfall/train/rainfall_10min_labeled.csv
seq_len: 24
n_classes: 4            # Overrides default 43 dataset-level classes to 4 GMM regimes

# --- Transform (Delay Embedding) ---
use_stft: false
delay: 8
embedding: 8

# --- Fine-Tuning & EMA ---
finetune: true
ft_method: all          # Fine-tune all U-Net and projection parameters
ema: true
ema_warmup: 100         # Warmup iterations for Exponential Moving Average weights
learning_rate: 0.0001   # AdamW learning rate (1e-4)
batch_size: 2048
epochs: 1001
```
