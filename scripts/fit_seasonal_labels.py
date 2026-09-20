#!/usr/bin/env bash
"""
Deterministic Seasonal-Phase Labeling & Holdout Split Generator
=============================================================================
Fits 4-level seasonal-phase conditioning index strictly on training years
(2000–2007) with zero holdout leakage, applies intra-day mode pooling and
minimum-duration dwell-time smoothing, and exports train/val/test splits
along with multi-scale block transition matrices.

Usage:
    python scripts/fit_seasonal_labels.py
"""

import os
import sys
import json
import hashlib
import pickle
from math import floor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'rainfall')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
os.makedirs(SPLITS_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, 'real_rainfall_data.csv')
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found!")
    sys.exit(1)

print(f"Loading raw dataset from {csv_path}...")
df_raw = pd.read_csv(csv_path)
df_raw['datetime'] = pd.to_datetime(df_raw['date'])
df_raw['year'] = df_raw['datetime'].dt.year
df_raw['annual_interval_idx'] = df_raw.groupby('year').cumcount()
df_raw['day_of_year'] = (df_raw['annual_interval_idx'] // 288) + 1
df_raw['rainfall_intensity'] = df_raw['avg_rainfall']

# Compute SHA-256
hasher = hashlib.sha256()
with open(csv_path, 'rb') as f:
    while chunk := f.read(65536):
        hasher.update(chunk)
raw_sha256 = hasher.hexdigest()

# 1. Partition splits
df_train = df_raw[(df_raw['year'] >= 2000) & (df_raw['year'] <= 2007)].copy()
df_val   = df_raw[df_raw['year'] == 2008].copy()
df_test  = df_raw[df_raw['year'] == 2009].copy()

# 2. Climatology profile on train split
clim_105120 = df_train.groupby('annual_interval_idx').agg(
    mean_intensity=('avg_rainfall', 'mean'),
    median_intensity=('avg_rainfall', 'median'),
    day_of_year=('day_of_year', 'first')
).reset_index().sort_values('annual_interval_idx')

# 14-day rolling mean
clim_105120['mean_intensity_smooth'] = clim_105120['mean_intensity'].rolling(
    window=4032, min_periods=1, center=True
).mean()

# 3. Fit 4-state HMM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clim_105120[['mean_intensity_smooth']])
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type='full', n_iter=200, random_state=42)
hmm_model.fit(X_scaled)
clim_105120['raw_state'] = hmm_model.predict(X_scaled)

# Order states monotonically
state_means = clim_105120.groupby('raw_state')['mean_intensity_smooth'].mean().sort_values()
state_order_map = {old_st: new_st for new_st, old_st in enumerate(state_means.index)}
clim_105120['hmm_state'] = clim_105120['raw_state'].map(state_order_map)

# 4. Daily Mode Pooling
clim_105120['day_idx'] = clim_105120['annual_interval_idx'] // 288
daily_modes = clim_105120.groupby('day_idx')['hmm_state'].agg(lambda s: s.value_counts().index[0])
clim_105120['hmm_state'] = clim_105120['day_idx'].map(daily_modes).astype(int)

# 5. Run-Length Encoding
states = clim_105120['hmm_state'].values
runs = []
cur_st = states[0]
st_idx = 0
for i in range(1, len(states)):
    if states[i] != cur_st or i == len(states) - 1:
        en_idx = i if states[i] != cur_st else i + 1
        runs.append({'state': cur_st, 'dur_steps': en_idx - st_idx, 'dur_days': (en_idx - st_idx) / 288.0})
        cur_st = states[i]
        st_idx = i
runs_df = pd.DataFrame(runs)
min_days_val = max(1, floor(runs_df['dur_days'].median()))

# 6. Minimum duration filter
def smooth_hmm_states_min_duration(state_seq, min_days=3, steps_per_day=288):
    min_steps = int(min_days * steps_per_day)
    s = state_seq.copy()
    changed = True
    while changed:
        changed = False
        runs = []
        cur_st = s[0]
        st_idx = 0
        for i in range(1, len(s)):
            if s[i] != cur_st or i == len(s) - 1:
                en_idx = i if s[i] != cur_st else i + 1
                runs.append((cur_st, st_idx, en_idx, en_idx - st_idx))
                cur_st = s[i]
                st_idx = i
        for idx, (st_val, start, end, dur) in enumerate(runs):
            if dur < min_steps:
                prev_st = runs[idx - 1][0] if idx > 0 else (runs[idx + 1][0] if idx < len(runs) - 1 else st_val)
                next_st = runs[idx + 1][0] if idx < len(runs) - 1 else prev_st
                if prev_st == next_st:
                    s[start:end] = prev_st
                else:
                    prev_dur = runs[idx - 1][3] if idx > 0 else 0
                    next_dur = runs[idx + 1][3] if idx < len(runs) - 1 else 0
                    s[start:end] = prev_st if prev_dur >= next_dur else next_st
                changed = True
                break
    return s

clim_105120['hmm_state_raw'] = clim_105120['hmm_state']
clim_105120['hmm_state'] = smooth_hmm_states_min_duration(
    clim_105120['hmm_state_raw'].values, 
    min_days=min_days_val
)
n_annual_trans = int(np.sum(clim_105120['hmm_state'].values[:-1] != clim_105120['hmm_state'].values[1:]))

# 7. Map to splits
interval_to_phase = dict(zip(clim_105120['annual_interval_idx'], clim_105120['hmm_state']))
for s_df in [df_train, df_val, df_test]:
    s_df['hmm_state'] = s_df['annual_interval_idx'].map(interval_to_phase).astype(int)
    s_df['date'] = s_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 8. Export CSVs
export_cols = ['datetime', 'date', 'avg_rainfall', 'rainfall_intensity', 'hmm_state', 'day_of_year', 'annual_interval_idx']
train_csv = os.path.join(SPLITS_DIR, 'train_years_labelled.csv')
val_csv   = os.path.join(SPLITS_DIR, 'val_years_labelled.csv')
test_csv  = os.path.join(SPLITS_DIR, 'test_years_labelled.csv')

df_train[export_cols].to_csv(train_csv, index=False)
df_val[export_cols].to_csv(val_csv, index=False)
df_test[export_cols].to_csv(test_csv, index=False)
print(f"Exported clean splits: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

# 9. Compute multi-scale transition matrices
smooth = 1e-5
train_states = df_train['hmm_state'].values
candidate_block_sizes = [1, 12, 24, 48, 72, 96, 144, 288]
by_block_size = {}

for bs in candidate_block_sizes:
    n_blocks = len(train_states) // bs
    block_seq = train_states.reshape(n_blocks, bs)[:, 0] if (288 % bs == 0) else train_states
    C_block = np.zeros((4, 4), dtype=np.float64) + smooth
    for i in range(len(block_seq) - 1):
        C_block[block_seq[i], block_seq[i + 1]] += 1.0
    P_block = C_block / C_block.sum(axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eig(P_block.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_block = np.real(eigenvectors[:, idx])
    pi_block = np.abs(pi_block / np.sum(pi_block))
    n_trans = sum(1 for i in range(len(block_seq) - 1) if block_seq[i] != block_seq[i + 1])
    
    by_block_size[bs] = {
        'P_block': P_block, 'P': P_block, 'C_block': C_block,
        'pi_block': pi_block, 'pi': pi_block, 'block_size': bs,
        'duration_hours': bs * 5 / 60.0, 'n_blocks': n_blocks,
        'n_transitions': n_trans, 'block_sequence': block_seq,
        'self_transition_diag': np.diag(P_block).tolist()
    }

unique, counts = np.unique(train_states, return_counts=True)
regime_proportions = {int(r): float(c / len(train_states)) for r, c in zip(unique, counts)}

unified_bundle = {
    'P_block': by_block_size[24]['P_block'],
    'P': by_block_size[24]['P'],
    'C_block': by_block_size[24]['C_block'],
    'pi_block': by_block_size[24]['pi_block'],
    'pi': by_block_size[24]['pi'],
    'block_size': 24,
    'block_sequence': by_block_size[24]['block_sequence'],
    'n_blocks': by_block_size[24]['n_blocks'],
    'n_transitions': by_block_size[24]['n_transitions'],
    'regime_proportions': regime_proportions,
    'n_states': 4,
    'post_processing_min_days': min_days_val,
    'by_block_size': by_block_size,
    'supported_block_sizes': candidate_block_sizes,
    'climatology_profile_1yr': clim_105120['hmm_state'].values,
    'calendar_sequence_1yr_step24': clim_105120['hmm_state'].values[::24],
    'calendar_sequence_1yr_step288': clim_105120['hmm_state'].values[::288],
}

main_pkl = os.path.join(SPLITS_DIR, 'seasonal_transition_matrix_train.pkl')
with open(main_pkl, 'wb') as f:
    pickle.dump(unified_bundle, f)

for bs in [24, 48, 72, 96, 144, 288]:
    standalone = {
        'P_block': by_block_size[bs]['P_block'],
        'P': by_block_size[bs]['P'],
        'C_block': by_block_size[bs]['C_block'],
        'pi_block': by_block_size[bs]['pi_block'],
        'pi': by_block_size[bs]['pi'],
        'block_size': bs,
        'duration_hours': by_block_size[bs]['duration_hours'],
        'block_sequence': by_block_size[bs]['block_sequence'],
        'n_blocks': by_block_size[bs]['n_blocks'],
        'n_transitions': by_block_size[bs]['n_transitions'],
        'regime_proportions': regime_proportions,
        'n_states': 4,
        'post_processing_min_days': min_days_val,
        'calendar_sequence_1yr': clim_105120['hmm_state'].values[::bs],
        'climatology_profile_1yr': clim_105120['hmm_state'].values,
    }
    with open(os.path.join(SPLITS_DIR, f'seasonal_transition_matrix_train_len{bs}.pkl'), 'wb') as f:
        pickle.dump(standalone, f)

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while c := f.read(65536):
            h.update(c)
    return h.hexdigest()

manifest = {
    'created_at': pd.Timestamp.now().isoformat(),
    'source_file': 'data/rainfall/real_rainfall_data.csv',
    'source_sha256': raw_sha256,
    'wet_threshold_mm': 0.005,
    'conditioning_mechanism': 'seasonal-phase-index',
    'post_processing': {
        'intra_day_mode_pooling': True,
        'min_duration_filter': 'smooth_hmm_states_min_duration',
        'min_days': min_days_val,
        'annual_transitions': n_annual_trans
    },
    'splits': {
        'train': {
            'file': 'train_years_labelled.csv',
            'years': '2000-2007',
            'rows': len(df_train),
            'sha256': get_sha256(train_csv),
            'phase_distribution': {int(k): int(v) for k, v in df_train['hmm_state'].value_counts().sort_index().items()}
        },
        'val': {
            'file': 'val_years_labelled.csv',
            'years': '2008',
            'rows': len(df_val),
            'sha256': get_sha256(val_csv),
            'phase_distribution': {int(k): int(v) for k, v in df_val['hmm_state'].value_counts().sort_index().items()}
        },
        'test': {
            'file': 'test_years_labelled.csv',
            'years': '2009',
            'rows': len(df_test),
            'sha256': get_sha256(test_csv),
            'phase_distribution': {int(k): int(v) for k, v in df_test['hmm_state'].value_counts().sort_index().items()}
        }
    },
    'transition_matrices': {
        'unified_bundle': 'seasonal_transition_matrix_train.pkl',
        'supported_block_sizes': candidate_block_sizes,
        'standalone_files': {
            f'len{bs}': f'seasonal_transition_matrix_train_len{bs}.pkl' for bs in [24, 48, 72, 96, 144, 288]
        }
    }
}

with open(os.path.join(SPLITS_DIR, 'MANIFEST.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print("✓ All holdout splits, multi-scale transition matrices, and MANIFEST.json successfully generated!")
