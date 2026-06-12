"""
Regime-Conditional Rainfall Dataset
====================================
Loads pre-split rainfall data (train_dataset.csv / test_dataset.csv)
with gmm_regime labels and creates sliding-window samples for training
the ImagenFew diffusion model.

Each block (identified by block_id) is a contiguous 33.6-hour segment of
rainfall at 5-minute resolution (4032 steps). This dataset downsamples
to 10-minute resolution (2016 steps per block) to match the existing
fine-tuned model, then extracts sliding windows of length seq_len.
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

    def __init__(self, csv_path, seq_len=24, scale=True, scaler=None):
        """
        Args:
            csv_path : Path to train_dataset.csv or test_dataset.csv
            seq_len  : Length of each sliding window
            scale    : Whether to apply StandardScaler normalization
            scaler   : Pre-fitted scaler (pass the train scaler when
                       building the test set).  If None, fits on this data.
        """
        self.seq_len = seq_len
        self.scale = scale

        df = pd.read_csv(csv_path)

        # Collect per-block arrays and regime labels
        blocks = []
        for _, group in df.groupby("block_id", sort=True):
            values = group["avg_rainfall"].values.astype(np.float32).reshape(-1, 1)
            regime = int(group["gmm_regime"].iloc[0])
            blocks.append((values, regime))

        # --- Scaler ---
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
