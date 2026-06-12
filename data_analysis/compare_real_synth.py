"""
Comprehensive Real vs Synthetic Rainfall Data Comparison
=========================================================
Evaluates whether synthetic rainfall data is fit for RL model training.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')
import os

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # project root
REAL_PATH = os.path.join(BASE_DIR, 'data/rainfall/real_rainfall_data.csv')
SYNTH_PATH = os.path.join(BASE_DIR, 'results/generated_data/rainfall_synthetic_10.0y_077efccb-d83d-4259-b120-6c04544f8fe0.csv')
OUT_DIR = os.path.join(BASE_DIR, 'results/comparison_report')
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load ---
df_real = pd.read_csv(REAL_PATH, parse_dates=['date'])
df_synth = pd.read_csv(SYNTH_PATH, parse_dates=['date'])

real = df_real['avg_rainfall'].values
synth = df_synth['avg_rainfall'].values.copy()
synth[synth < 0.005] = 0.0

print("=" * 70)
print("REAL vs SYNTHETIC RAINFALL — COMPREHENSIVE COMPARISON")
print("=" * 70)

# =========================================================================
# 1. BASIC STATISTICS
# =========================================================================
print("\n" + "=" * 70)
print("1. BASIC STATISTICS")
print("=" * 70)

def stats_summary(arr, label):
    return {
        'Dataset': label,
        'N': len(arr),
        'Mean': np.mean(arr),
        'Std': np.std(arr),
        'Min': np.min(arr),
        'Max': np.max(arr),
        'Median': np.median(arr),
        'P90': np.percentile(arr, 90),
        'P95': np.percentile(arr, 95),
        'P99': np.percentile(arr, 99),
        'P99.9': np.percentile(arr, 99.9),
        'Skewness': stats.skew(arr),
        'Kurtosis': stats.kurtosis(arr),
        'Zero Fraction': np.mean(arr == 0),
        'Near-Zero Fraction (< 0.01)': np.mean(arr < 0.01),
    }

s_real = stats_summary(real, 'Real')
s_synth = stats_summary(synth, 'Synthetic')

df_stats = pd.DataFrame([s_real, s_synth]).set_index('Dataset').T
print(df_stats.to_string())

# =========================================================================
# 2. DISTRIBUTION COMPARISON
# =========================================================================
print("\n" + "=" * 70)
print("2. DISTRIBUTION COMPARISON")
print("=" * 70)

# KS test on full distributions
ks_stat, ks_p = stats.ks_2samp(real, synth)
print(f"  KS Statistic (full):      {ks_stat:.6f}")
print(f"  KS p-value (full):        {ks_p:.2e}")

# KS test on non-zero values only (wet periods)
real_nz = real[real > 0]
synth_nz = synth[synth > 0]
ks_nz_stat, ks_nz_p = stats.ks_2samp(real_nz, synth_nz)
print(f"  KS Statistic (wet only):  {ks_nz_stat:.6f}")
print(f"  KS p-value (wet only):    {ks_nz_p:.2e}")

# Jensen-Shannon Divergence on intensity distributions
bins = np.linspace(0, max(real.max(), synth.max()), 300)
hist_real, _ = np.histogram(real, bins=bins, density=True)
hist_synth, _ = np.histogram(synth, bins=bins, density=True)
# Add small epsilon to avoid log(0)
hist_real = hist_real + 1e-12
hist_synth = hist_synth + 1e-12
jsd = jensenshannon(hist_real, hist_synth)
print(f"  Jensen-Shannon Divergence: {jsd:.6f}  (0=identical, 1=maximally different)")

# JSD on wet-only
bins_nz = np.linspace(0.001, max(real_nz.max(), synth_nz.max()), 200)
hist_real_nz, _ = np.histogram(real_nz, bins=bins_nz, density=True)
hist_synth_nz, _ = np.histogram(synth_nz, bins=bins_nz, density=True)
hist_real_nz = hist_real_nz + 1e-12
hist_synth_nz = hist_synth_nz + 1e-12
jsd_nz = jensenshannon(hist_real_nz, hist_synth_nz)
print(f"  JSD (wet-only):            {jsd_nz:.6f}")

# --- Plot: Distribution comparison ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Full distribution (log scale)
axes[0].hist(real, bins=200, density=True, alpha=0.5, label='Real', color='steelblue')
axes[0].hist(synth, bins=200, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[0].set_yscale('log')
axes[0].set_title('Full Distribution (log-scale)')
axes[0].set_xlabel('Rainfall Intensity (mm)')
axes[0].legend()

# Wet-only distribution
axes[1].hist(real_nz, bins=150, density=True, alpha=0.5, label='Real (wet)', color='steelblue')
axes[1].hist(synth_nz, bins=150, density=True, alpha=0.5, label='Synthetic (wet)', color='coral')
axes[1].set_yscale('log')
axes[1].set_title('Wet-Period Distribution (log-scale)')
axes[1].set_xlabel('Rainfall Intensity (mm)')
axes[1].legend()

# QQ plot (wet-only)
real_nz_sorted = np.sort(real_nz)
synth_nz_sorted = np.sort(synth_nz)
# Subsample to same length for QQ
n_qq = min(len(real_nz_sorted), len(synth_nz_sorted), 5000)
q_real = np.quantile(real_nz_sorted, np.linspace(0, 1, n_qq))
q_synth = np.quantile(synth_nz_sorted, np.linspace(0, 1, n_qq))
axes[2].scatter(q_real, q_synth, s=2, alpha=0.5, color='purple')
max_val = max(q_real.max(), q_synth.max())
axes[2].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x (ideal)')
axes[2].set_title('Q-Q Plot (Wet Periods)')
axes[2].set_xlabel('Real Quantiles')
axes[2].set_ylabel('Synthetic Quantiles')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_distribution_comparison.png'), dpi=150)
plt.close()
print(f"  [Saved] 01_distribution_comparison.png")

# =========================================================================
# 3. WET/DRY SPELL ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("3. WET/DRY SPELL ANALYSIS")
print("=" * 70)

def extract_spells(arr, threshold=0):
    """Extract consecutive wet and dry spell durations."""
    is_wet = (arr > threshold).astype(int)
    changes = np.diff(is_wet, prepend=-1)
    spell_starts = np.where(changes != 0)[0]
    spell_lengths = np.diff(np.append(spell_starts, len(arr)))
    spell_types = is_wet[spell_starts]  # 1=wet, 0=dry
    wet_spells = spell_lengths[spell_types == 1]
    dry_spells = spell_lengths[spell_types == 0]
    return wet_spells, dry_spells

wet_real, dry_real = extract_spells(real)
wet_synth, dry_synth = extract_spells(synth)

print(f"  {'':30s} {'Real':>12s} {'Synthetic':>12s}")
print(f"  {'Num wet spells':30s} {len(wet_real):12d} {len(wet_synth):12d}")
print(f"  {'Num dry spells':30s} {len(dry_real):12d} {len(dry_synth):12d}")
print(f"  {'Mean wet spell (steps)':30s} {wet_real.mean():12.2f} {wet_synth.mean():12.2f}")
print(f"  {'Median wet spell (steps)':30s} {np.median(wet_real):12.1f} {np.median(wet_synth):12.1f}")
print(f"  {'Max wet spell (steps)':30s} {wet_real.max():12d} {wet_synth.max():12d}")
print(f"  {'Mean dry spell (steps)':30s} {dry_real.mean():12.2f} {dry_synth.mean():12.2f}")
print(f"  {'Median dry spell (steps)':30s} {np.median(dry_real):12.1f} {np.median(dry_synth):12.1f}")
print(f"  {'Max dry spell (steps)':30s} {dry_real.max():12d} {dry_synth.max():12d}")
print(f"  (1 step = 10 minutes)")

# KS test on spell distributions
ks_wet, p_wet = stats.ks_2samp(wet_real, wet_synth)
ks_dry, p_dry = stats.ks_2samp(dry_real, dry_synth)
print(f"\n  KS test on wet spells:  stat={ks_wet:.4f}, p={p_wet:.2e}")
print(f"  KS test on dry spells:  stat={ks_dry:.4f}, p={p_dry:.2e}")

# --- Plot: Spell distributions ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

max_wet = int(min(np.percentile(wet_real, 99), np.percentile(wet_synth, 99)) * 1.5)
bins_wet = np.arange(0, max_wet + 1)
axes[0].hist(wet_real, bins=bins_wet, density=True, alpha=0.5, label='Real', color='steelblue')
axes[0].hist(wet_synth, bins=bins_wet, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[0].set_title('Wet Spell Duration Distribution')
axes[0].set_xlabel('Duration (steps × 10 min)')
axes[0].set_ylabel('Density')
axes[0].legend()

max_dry = int(min(np.percentile(dry_real, 99), np.percentile(dry_synth, 99)) * 1.5)
bins_dry = np.arange(0, max_dry + 1, max(1, max_dry // 50))
axes[1].hist(dry_real, bins=bins_dry, density=True, alpha=0.5, label='Real', color='steelblue')
axes[1].hist(dry_synth, bins=bins_dry, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[1].set_title('Dry Spell Duration Distribution')
axes[1].set_xlabel('Duration (steps × 10 min)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_spell_analysis.png'), dpi=150)
plt.close()
print(f"  [Saved] 02_spell_analysis.png")

# =========================================================================
# 4. TEMPORAL AUTOCORRELATION
# =========================================================================
print("\n" + "=" * 70)
print("4. TEMPORAL AUTOCORRELATION")
print("=" * 70)

max_lag = 144  # 24 hours at 10-min intervals

def compute_acf(arr, max_lag):
    n = len(arr)
    mean = np.mean(arr)
    var = np.var(arr)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.mean((arr[:n-lag] - mean) * (arr[lag:] - mean)) / var if var > 0 else 0
    return acf

acf_real = compute_acf(real, max_lag)
acf_synth = compute_acf(synth, max_lag)

# ACF error
acf_mae = np.mean(np.abs(acf_real - acf_synth))
acf_rmse = np.sqrt(np.mean((acf_real - acf_synth)**2))
print(f"  ACF MAE  (lag 0-{max_lag}): {acf_mae:.6f}")
print(f"  ACF RMSE (lag 0-{max_lag}): {acf_rmse:.6f}")

fig, ax = plt.subplots(figsize=(12, 5))
lags_hours = np.arange(max_lag + 1) * 10 / 60  # convert to hours
ax.plot(lags_hours, acf_real, label='Real', color='steelblue', linewidth=2)
ax.plot(lags_hours, acf_synth, label='Synthetic', color='coral', linewidth=2)
ax.set_title('Autocorrelation Function (ACF)')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Autocorrelation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_autocorrelation.png'), dpi=150)
plt.close()
print(f"  [Saved] 03_autocorrelation.png")

# =========================================================================
# 5. EXTREME VALUE ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("5. EXTREME VALUE ANALYSIS")
print("=" * 70)

percentiles = [90, 95, 99, 99.5, 99.9]
print(f"  {'Percentile':>12s} {'Real':>12s} {'Synthetic':>12s} {'Ratio(S/R)':>12s}")
for p in percentiles:
    r_val = np.percentile(real, p)
    s_val = np.percentile(synth, p)
    ratio = s_val / r_val if r_val > 0 else float('inf')
    print(f"  {p:>11.1f}% {r_val:12.4f} {s_val:12.4f} {ratio:12.4f}")

# Exceedance probability plot
sorted_real_nz = np.sort(real_nz)[::-1]
sorted_synth_nz = np.sort(synth_nz)[::-1]
prob_real = np.arange(1, len(sorted_real_nz) + 1) / len(sorted_real_nz)
prob_synth = np.arange(1, len(sorted_synth_nz) + 1) / len(sorted_synth_nz)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(sorted_real_nz, prob_real, label='Real', color='steelblue', linewidth=1.5)
ax.semilogy(sorted_synth_nz, prob_synth, label='Synthetic', color='coral', linewidth=1.5)
ax.set_title('Exceedance Probability (Wet Periods)')
ax.set_xlabel('Rainfall Intensity (mm)')
ax.set_ylabel('P(X > x)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_exceedance_probability.png'), dpi=150)
plt.close()
print(f"  [Saved] 04_exceedance_probability.png")

# =========================================================================
# 6. HOURLY / DAILY AGGREGATION COMPARISON
# =========================================================================
print("\n" + "=" * 70)
print("6. HOURLY AGGREGATION COMPARISON")
print("=" * 70)

# Aggregate to hourly (6 steps = 1 hour)
n_hours_real = len(real) // 6
n_hours_synth = len(synth) // 6
hourly_real = real[:n_hours_real * 6].reshape(-1, 6).sum(axis=1)
hourly_synth = synth[:n_hours_synth * 6].reshape(-1, 6).sum(axis=1)

ks_hourly, p_hourly = stats.ks_2samp(hourly_real, hourly_synth)
print(f"  Hourly KS stat: {ks_hourly:.6f}, p={p_hourly:.2e}")
print(f"  Hourly mean  — Real: {hourly_real.mean():.4f}, Synth: {hourly_synth.mean():.4f}")
print(f"  Hourly std   — Real: {hourly_real.std():.4f}, Synth: {hourly_synth.std():.4f}")
print(f"  Hourly max   — Real: {hourly_real.max():.4f}, Synth: {hourly_synth.max():.4f}")

# Daily (144 steps = 1 day)
n_days_real = len(real) // 144
n_days_synth = len(synth) // 144
daily_real = real[:n_days_real * 144].reshape(-1, 144).sum(axis=1)
daily_synth = synth[:n_days_synth * 144].reshape(-1, 144).sum(axis=1)

ks_daily, p_daily = stats.ks_2samp(daily_real, daily_synth)
print(f"\n  Daily KS stat:  {ks_daily:.6f}, p={p_daily:.2e}")
print(f"  Daily mean   — Real: {daily_real.mean():.4f}, Synth: {daily_synth.mean():.4f}")
print(f"  Daily std    — Real: {daily_real.std():.4f}, Synth: {daily_synth.std():.4f}")
print(f"  Daily max    — Real: {daily_real.max():.4f}, Synth: {daily_synth.max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(hourly_real[hourly_real > 0], bins=100, density=True, alpha=0.5, label='Real', color='steelblue')
axes[0].hist(hourly_synth[hourly_synth > 0], bins=100, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[0].set_title('Hourly Rainfall Distribution (non-zero)')
axes[0].set_xlabel('Hourly Rainfall (mm)')
axes[0].set_yscale('log')
axes[0].legend()

axes[1].hist(daily_real[daily_real > 0], bins=80, density=True, alpha=0.5, label='Real', color='steelblue')
axes[1].hist(daily_synth[daily_synth > 0], bins=80, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[1].set_title('Daily Rainfall Distribution (non-zero)')
axes[1].set_xlabel('Daily Rainfall (mm)')
axes[1].set_yscale('log')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_aggregated_distributions.png'), dpi=150)
plt.close()
print(f"  [Saved] 05_aggregated_distributions.png")

# =========================================================================
# 7. STORM EVENT ANALYSIS (RL-relevant)
# =========================================================================
print("\n" + "=" * 70)
print("7. STORM EVENT ANALYSIS (RL-Critical)")
print("=" * 70)

def extract_storms(arr, min_duration=3):
    is_wet = (arr > 0).astype(int)
    changes = np.diff(is_wet, prepend=-1)
    spell_starts = np.where(changes != 0)[0]
    spell_lengths = np.diff(np.append(spell_starts, len(arr)))
    spell_types = is_wet[spell_starts]
    storms = []
    for i, (start, length, stype) in enumerate(zip(spell_starts, spell_lengths, spell_types)):
        if stype == 1 and length >= min_duration:
            storms.append(arr[start:start + length])
    return storms

storms_real = extract_storms(real)
storms_synth = extract_storms(synth)

dur_real = np.array([len(s) for s in storms_real])
dur_synth = np.array([len(s) for s in storms_synth])
peak_real = np.array([s.max() for s in storms_real])
peak_synth = np.array([s.max() for s in storms_synth])
total_real = np.array([s.sum() for s in storms_real])
total_synth = np.array([s.sum() for s in storms_synth])
peak_pos_real = np.array([np.argmax(s) / len(s) for s in storms_real])
peak_pos_synth = np.array([np.argmax(s) / len(s) for s in storms_synth])

print(f"  {'':35s} {'Real':>12s} {'Synthetic':>12s}")
print(f"  {'Number of storms':35s} {len(storms_real):12d} {len(storms_synth):12d}")
print(f"  {'Mean duration (steps)':35s} {dur_real.mean():12.2f} {dur_synth.mean():12.2f}")
print(f"  {'Mean peak intensity':35s} {peak_real.mean():12.4f} {peak_synth.mean():12.4f}")
print(f"  {'Max peak intensity':35s} {peak_real.max():12.4f} {peak_synth.max():12.4f}")
print(f"  {'Mean total storm volume':35s} {total_real.mean():12.4f} {total_synth.mean():12.4f}")
print(f"  {'Mean peak position (0=start)':35s} {peak_pos_real.mean():12.4f} {peak_pos_synth.mean():12.4f}")

ks_dur, p_dur = stats.ks_2samp(dur_real, dur_synth)
ks_peak, p_peak = stats.ks_2samp(peak_real, peak_synth)
ks_vol, p_vol = stats.ks_2samp(total_real, total_synth)
print(f"\n  KS test (storm duration):  stat={ks_dur:.4f}, p={p_dur:.2e}")
print(f"  KS test (peak intensity):  stat={ks_peak:.4f}, p={p_peak:.2e}")
print(f"  KS test (storm volume):    stat={ks_vol:.4f}, p={p_vol:.2e}")

# --- Plot: Storm characteristics ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(dur_real * 10, bins=50, density=True, alpha=0.5, label='Real', color='steelblue')
axes[0, 0].hist(dur_synth * 10, bins=50, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[0, 0].set_title('Storm Duration Distribution')
axes[0, 0].set_xlabel('Duration (minutes)')
axes[0, 0].legend()

axes[0, 1].hist(peak_real, bins=50, density=True, alpha=0.5, label='Real', color='steelblue')
axes[0, 1].hist(peak_synth, bins=50, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[0, 1].set_title('Storm Peak Intensity Distribution')
axes[0, 1].set_xlabel('Peak Intensity (mm)')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()

axes[1, 0].hist(total_real, bins=50, density=True, alpha=0.5, label='Real', color='steelblue')
axes[1, 0].hist(total_synth, bins=50, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[1, 0].set_title('Storm Total Volume Distribution')
axes[1, 0].set_xlabel('Total Volume (mm)')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()

axes[1, 1].hist(peak_pos_real, bins=30, density=True, alpha=0.5, label='Real', color='steelblue')
axes[1, 1].hist(peak_pos_synth, bins=30, density=True, alpha=0.5, label='Synthetic', color='coral')
axes[1, 1].set_title('Peak Timing within Storm')
axes[1, 1].set_xlabel('Relative Position (0=start, 1=end)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_storm_event_analysis.png'), dpi=150)
plt.close()
print(f"  [Saved] 06_storm_event_analysis.png")

# =========================================================================
# 8. SAMPLE TIMESERIES VISUAL COMPARISON
# =========================================================================
print("\n" + "=" * 70)
print("8. SAMPLE TIMESERIES VISUAL COMPARISON")
print("=" * 70)

# Show 7 days of data from each
window = 144 * 7  # 7 days

# Find an interesting window in real data (one with some rain)
for start in range(0, len(real) - window, 144):
    if real[start:start+window].sum() > 5:
        break
real_window = real[start:start+window]

for start in range(0, len(synth) - window, 144):
    if synth[start:start+window].sum() > 5:
        break
synth_window = synth[start:start+window]

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
hours = np.arange(window) * 10 / 60

axes[0].fill_between(hours, real_window, alpha=0.7, color='steelblue')
axes[0].set_title('Real Rainfall — 7-day Sample')
axes[0].set_ylabel('Intensity (mm)')

axes[1].fill_between(hours, synth_window, alpha=0.7, color='coral')
axes[1].set_title('Synthetic Rainfall — 7-day Sample')
axes[1].set_ylabel('Intensity (mm)')
axes[1].set_xlabel('Time (hours)')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_timeseries_sample.png'), dpi=150)
plt.close()
print(f"  [Saved] 07_timeseries_sample.png")

# =========================================================================
# 9. RL-FITNESS SCORECARD
# =========================================================================
print("\n" + "=" * 70)
print("9. RL-FITNESS SCORECARD")
print("=" * 70)

# Compute scores for each dimension
scores = {}

# a) Wet/dry ratio match
wet_ratio_real = np.mean(real > 0)
wet_ratio_synth = np.mean(synth > 0)
wet_ratio_error = abs(wet_ratio_real - wet_ratio_synth) / wet_ratio_real
scores['Wet/Dry Ratio'] = max(0, 1 - wet_ratio_error * 5)  # penalize 20% per 1x relative error

# b) Mean intensity match (wet-only)
mean_err = abs(real_nz.mean() - synth_nz.mean()) / real_nz.mean()
scores['Mean Intensity (wet)'] = max(0, 1 - mean_err * 2)

# c) Distribution match (JSD)
scores['Distribution (JSD)'] = max(0, 1 - jsd_nz * 4)  # penalize heavily for divergence

# d) Autocorrelation match
scores['Autocorrelation'] = max(0, 1 - acf_mae * 10)

# e) Storm duration match
dur_err = abs(dur_real.mean() - dur_synth.mean()) / dur_real.mean()
scores['Storm Duration'] = max(0, 1 - dur_err * 3)

# f) Extreme values (99th percentile ratio)
extreme_ratio = np.percentile(synth_nz, 99) / np.percentile(real_nz, 99)
scores['Extreme Values (P99)'] = max(0, 1 - abs(1 - extreme_ratio) * 3)

# g) Storm count ratio
count_ratio = len(storms_synth) / len(storms_real)
scores['Storm Count'] = max(0, 1 - abs(1 - count_ratio) * 2)

# h) Peak timing
peak_pos_err = abs(peak_pos_real.mean() - peak_pos_synth.mean())
scores['Peak Timing'] = max(0, 1 - peak_pos_err * 5)

print(f"\n  {'Dimension':35s} {'Score':>8s}  {'Status':>10s}")
print(f"  {'-'*55}")
for dim, score in scores.items():
    status = '✅ PASS' if score >= 0.7 else ('⚠️  WARN' if score >= 0.4 else '❌ FAIL')
    bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
    print(f"  {dim:35s} {score:8.3f}  {bar} {status}")

overall = np.mean(list(scores.values()))
print(f"\n  {'OVERALL SCORE':35s} {overall:8.3f}")

if overall >= 0.7:
    verdict = "✅ FIT FOR RL TRAINING"
    detail = "The synthetic data captures the key statistical properties of the real rainfall data well enough for RL training."
elif overall >= 0.5:
    verdict = "⚠️  CONDITIONALLY FIT"
    detail = "The synthetic data has notable deviations in some dimensions. RL training may work but could learn biased policies."
else:
    verdict = "❌ NOT FIT FOR RL TRAINING"
    detail = "The synthetic data deviates significantly from real data. RL policies trained on this may not transfer to real conditions."

print(f"\n  VERDICT: {verdict}")
print(f"  {detail}")

# Specific RL concerns
print(f"\n  --- RL-SPECIFIC CONCERNS ---")
print(f"  Wet/Dry Ratio — Real: {wet_ratio_real:.4f}, Synth: {wet_ratio_synth:.4f}")
print(f"    → {'OK' if scores['Wet/Dry Ratio'] >= 0.7 else 'CONCERN'}: RL agent will {'see similar' if scores['Wet/Dry Ratio'] >= 0.7 else 'see different'} frequency of rain events.")
print(f"  Storm Duration — Mean Real: {dur_real.mean()*10:.0f}min, Synth: {dur_synth.mean()*10:.0f}min")
print(f"    → {'OK' if scores['Storm Duration'] >= 0.7 else 'CONCERN'}: Episode lengths will {'match' if scores['Storm Duration'] >= 0.7 else 'differ from'} real conditions.")
print(f"  Extreme Events — P99 Ratio: {extreme_ratio:.3f}")
print(f"    → {'OK' if scores['Extreme Values (P99)'] >= 0.7 else 'CONCERN'}: RL agent will {'encounter similar' if scores['Extreme Values (P99)'] >= 0.7 else 'not encounter realistic'} extreme rainfall.")

print("\n" + "=" * 70)
print("REPORT COMPLETE — Plots saved to:", OUT_DIR)
print("=" * 70)
