"""
Regime-Conditional Rainfall Dataset
====================================
Loads pre-split rainfall data (train_dataset.csv / test_dataset.csv)
with regime labels (GMM or HMM) and creates sliding-window samples for
training the ImagenFew diffusion model.

Supports configurable column names:
  - regime_col: 'gmm_regime' (default) or 'hmm_state'
  - data_col:   'avg_rainfall' (default) or 'rainfall_intensity'
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class RegimeDataset(Dataset):
    """
    Dataset returning (timeseries_window, regime_label) pairs.

    - timeseries_window: [seq_len, 1] tensor of scaled rainfall values
    - regime_label: integer in {0, 1, 2, 3}
    """

    def __init__(self, csv_path, seq_len=24, scale=True, scaler=None,
                 regime_col="gmm_regime", data_col="avg_rainfall",
                 block_col="block_id"):
        """
        Args:
            csv_path   : Path to train or test CSV
            seq_len    : Length of each sliding window
            scale      : Whether to apply StandardScaler normalization
            scaler     : Pre-fitted scaler (pass the train scaler when
                        building the test set).  If None, fits on this data.
            regime_col : Column name for regime/state labels
                        ('gmm_regime' for GMM, 'hmm_state' for HMM)
            data_col   : Column name for rainfall values
                        ('avg_rainfall' or 'rainfall_intensity')
            block_col  : Column name for block grouping.
                        If not present, uses contiguous runs of the same
                        regime as implicit blocks.
        """
        self.seq_len = seq_len
        self.scale = scale
        self.regime_col = regime_col
        self.data_col = data_col

        df = pd.read_csv(csv_path)

        # Validate required columns
        if regime_col not in df.columns:
            raise ValueError(
                f"Regime column '{regime_col}' not found. "
                f"Available: {df.columns.tolist()}"
            )
        if data_col not in df.columns:
            raise ValueError(
                f"Data column '{data_col}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        # Collect per-block arrays and regime labels
        blocks = []
        if block_col in df.columns:
            # Grouped by explicit block_id
            for _, group in df.groupby(block_col, sort=True):
                values = group[data_col].values.astype(np.float32).reshape(-1, 1)
                regime = int(group[regime_col].iloc[0])
                blocks.append((values, regime))
        else:
            # No block_id → treat entire dataset as one continuous block
            # and create sliding windows directly with per-window regime
            # assignment (majority state in each window)
            values = df[data_col].values.astype(np.float32).reshape(-1, 1)
            regimes = df[regime_col].values.astype(int)

            # --- Scaler ---
            if scale:
                if scaler is not None:
                    self.scaler = scaler
                else:
                    self.scaler = StandardScaler()
                    self.scaler.fit(values)
                scaled = self.scaler.transform(values)
            else:
                self.scaler = None
                scaled = values

            # --- Sliding windows (continuous mode) ---
            self.samples = []
            self.labels = []
            n_windows = len(scaled) - seq_len + 1
            for i in range(n_windows):
                self.samples.append(
                    torch.FloatTensor(scaled[i : i + seq_len])
                )
                # Majority state in the window
                window_states = regimes[i : i + seq_len]
                vals, counts = np.unique(window_states, return_counts=True)
                self.labels.append(int(vals[np.argmax(counts)]))

            self.labels = torch.LongTensor(self.labels)
            self.n_regimes = 4  # Fixed: 0, 1, 2, 3
            return

        # --- Block-based path (original GMM flow) ---
        if scale:
            all_values = np.concatenate([b[0] for b in blocks], axis=0)
            if scaler is not None:
                self.scaler = scaler
            else:
                self.scaler = StandardScaler()
                self.scaler.fit(all_values)
        else:
            self.scaler = None

        # --- Sliding windows ---
        self.samples = []
        self.labels = []

        for values, regime in blocks:
            scaled = self.scaler.transform(values) if scale else values
            n_windows = len(scaled) - seq_len + 1
            for i in range(n_windows):
                self.samples.append(torch.FloatTensor(scaled[i : i + seq_len]))
                self.labels.append(regime)

        self.labels = torch.LongTensor(self.labels)
        self.n_regimes = 4  # Fixed: 0, 1, 2, 3

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_regime_counts(self):
        """Return {regime_id: sample_count}."""
        unique, counts = self.labels.unique(return_counts=True)
        return {int(r): int(c) for r, c in zip(unique, counts)}

