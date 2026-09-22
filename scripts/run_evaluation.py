"""
run_evaluation.py  —  compute descriptive sanity metrics for v8/v9/v10 and print
structured results.

CAVEATS (added 2026-09-22, see code_plan/AUDIT_2026-09-22.md §5 items 5 and 8):

1. REAL_PATH below is the full 2000-2009 record, 80% of which is v8/v9/v10's own
   training data. `my notes/memo/day_1.md` §3.1 freezes the 2000-2007 TRAINING
   PARTITION as the canonical Gate A reference, and designates full-record values as
   a documented sensitivity result, not a pass/fail target. Verdicts computed here can
   differ materially from the canonical ones (e.g. v10's hourly ACF RMSE is 0.055
   FAIL against this file's reference vs 0.049 PASS against the canonical one).
   Use `scripts/gate_a_scorecard.py --reference train` (or `--reference test` for the
   locked 2009 year) for any number that will be reported as a result.
2. "P99"/"P99.9" below are computed over the FULL series including zeros (e.g. real
   P99 = 0.160 mm). `code_plan/ACCEPTANCE_CRITERIA.md` and
   `code_plan/EVALUATION_GUIDE.md` use the same names for the WET-ONLY quantile
   (real P99_wet = 0.595 mm) — a different number. Do not compare the two.
3. This script has no machine-readable pass/fail column and does not compute Tier 4
   (IDF) or Tier 5 (seasonality/diurnal). `scripts/gate_a_scorecard.py` covers all of
   Gate A as specified in `code_plan/ACCEPTANCE_CRITERIA.md`, with explicit bands.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR   = os.path.join(PROJECT_ROOT, 'results', 'generated_data')
REAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'rainfall', 'real_rainfall_data.csv')  # full 2000-2009 record — see caveat 1 above
VERSIONS  = ['v8','v8_cal','v9','v9_cal','v10','v10_cal']

# ── Load ────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path, parse_dates=['date'])
    col = 'avg_rainfall' if 'avg_rainfall' in df.columns else df.columns[1]
    df = df[['date', col]].rename(columns={col: 'avg_rainfall'})
    df['avg_rainfall'] = df['avg_rainfall'].clip(lower=0)
    df.loc[df['avg_rainfall'] < 0.005, 'avg_rainfall'] = 0.0
    return df

DATA = {'Real': load(REAL_PATH)}
for v in VERSIONS:
    p = os.path.join(GEN_DIR, f'rainfall_synthetic_10y_{v}.csv')
    if os.path.exists(p):
        DATA[v] = load(p)
    else:
        print(f'MISSING: {p}')

NAMES = ['Real'] + [v for v in VERSIONS if v in DATA]
SYN   = [v for v in VERSIONS if v in DATA]

# ── Helpers ─────────────────────────────────────────────────────────────
def arr(n):  return DATA[n]['avg_rainfall'].values
def nz(n):   a = arr(n); return a[a > 0]
def to_hourly(a): n = len(a)//12*12;   return a[:n].reshape(-1, 12).sum(1)
def to_daily(a):  n = len(a)//288*288; return a[:n].reshape(-1, 288).sum(1)
def n_years(n):   return len(arr(n)) / (365 * 288)

def compute_acf(a, max_lag):
    m, v = a.mean(), a.var()
    if v == 0: return np.zeros(max_lag + 1)
    N = len(a)
    return np.array([np.mean((a[:N-k]-m)*(a[k:]-m))/v for k in range(max_lag + 1)])

def extract_spells(a):
    wet = (a > 0).astype(int)
    if wet.sum() == 0: return np.array([0]), np.array([len(a)])
    ch = np.diff(wet, prepend=-1); starts = np.where(ch != 0)[0]
    lengths = np.diff(np.append(starts, len(a))); types = wet[starts]
    return lengths[types==1], lengths[types==0]

def extract_storms(a, min_steps=3):
    wet = (a > 0).astype(int); ch = np.diff(wet, prepend=-1)
    starts = np.where(ch != 0)[0]; lengths = np.diff(np.append(starts, len(a))); types = wet[starts]
    return [a[s:s+l] for s,l,t in zip(starts, lengths, types) if t==1 and l>=min_steps]

sep = "=" * 76

# ── 1. Annual Volume ─────────────────────────────────────────────────────
print(sep); print("1. WATER BALANCE — Annual Volume"); print(sep)
ann_vol  = {n: arr(n).sum() / n_years(n) for n in NAMES}
real_vol = ann_vol['Real']
for n in NAMES:
    r    = ann_vol[n] / real_vol
    flag = '✅' if 0.90<=r<=1.10 else ('⚠️' if 0.80<=r<=1.20 else '❌')
    print(f"  {n:<10} {ann_vol[n]:>9.2f} mm/yr   ratio={r:.4f}  {flag}")

# ── 2. Intermittency ────────────────────────────────────────────────────
print(f"\n{sep}"); print("2. INTERMITTENCY — Zero Fraction & Spells"); print(sep)
spell_stats = {}
for n in NAMES:
    a = arr(n); ws, ds = extract_spells(a)
    spell_stats[n] = dict(
        zero_pct     = (a==0).mean()*100,
        mean_wet_min = ws.mean()*5,    med_wet_min=np.median(ws)*5,
        p95_wet_hr   = np.percentile(ws, 95)*5/60,
        max_wet_hr   = ws.max()*5/60,
        mean_dry_hr  = ds.mean()*5/60, med_dry_hr=np.median(ds)*5/60,
        p95_dry_hr   = np.percentile(ds, 95)*5/60,
        n_wet=len(ws), n_dry=len(ds),
        wet_spells=ws, dry_spells=ds)
    s = spell_stats[n]
    print(f"  {n:<10} zero={s['zero_pct']:.2f}%  wet_mean={s['mean_wet_min']:.1f}min  "
          f"wet_med={s['med_wet_min']:.1f}min  wet_p95={s['p95_wet_hr']:.2f}hr  "
          f"dry_mean={s['mean_dry_hr']:.2f}hr  dry_med={s['med_dry_hr']:.2f}hr  "
          f"dry_p95={s['p95_dry_hr']:.1f}hr  n_events={s['n_wet']}")

# ── 3. Marginal distribution ─────────────────────────────────────────────
print(f"\n{sep}"); print("3. MARGINAL INTENSITY DISTRIBUTION"); print(sep)
real_h = to_hourly(arr('Real')); real_d = to_daily(arr('Real'))
print(f"  {'Version':<10} {'MeanWet':>9} {'StdWet':>9} {'P90':>8} {'P95':>8} {'P99':>8} {'P99.9':>8} {'Max':>8} {'KS_h':>8} {'KS_d':>8}")
for n in NAMES:
    a  = arr(n); h = to_hourly(a); d = to_daily(a)
    nzv= nz(n)
    ks_h = stats.ks_2samp(real_h, h).statistic if n!='Real' else 0.0
    ks_d = stats.ks_2samp(real_d, d).statistic if n!='Real' else 0.0
    print(f"  {n:<10} {nzv.mean():>9.5f} {nzv.std():>9.5f} "
          f"{np.percentile(a,90):>8.4f} {np.percentile(a,95):>8.4f} "
          f"{np.percentile(a,99):>8.4f} {np.percentile(a,99.9):>8.4f} "
          f"{a.max():>8.4f} {ks_h:>8.4f} {ks_d:>8.4f}")

# ── 4. Autocorrelation ───────────────────────────────────────────────────
print(f"\n{sep}"); print("4. AUTOCORRELATION"); print(sep)
real_acf_h = compute_acf(real_h, 24)
for n in NAMES:
    a = arr(n); h = to_hourly(a)
    acf_n = compute_acf(a, 12)
    acf_h = compute_acf(h, 24)
    rmse  = float(np.sqrt(np.mean((real_acf_h - acf_h)**2))) if n!='Real' else 0.0
    print(f"  {n:<10} lag1_5min={acf_n[1]:.4f}  lag2={acf_n[2]:.4f}  lag6={acf_n[6]:.4f}  "
          f"lag1_hr={acf_h[1]:.4f}  lag6_hr={acf_h[6]:.4f}  lag12_hr={acf_h[12]:.4f}  acf_rmse={rmse:.4f}")

# ── 5. Storm properties ──────────────────────────────────────────────────
print(f"\n{sep}"); print("5. STORM EVENT PROPERTIES"); print(sep)
storm_data = {}
for n in NAMES:
    a = arr(n); ny = n_years(n); st = extract_storms(a, 3)
    if not st:
        storm_data[n] = {k:0 for k in ['n_per_yr','mean_dur','p50_dur','p95_dur','max_dur',
                                         'mean_peak','p99_peak','max_peak','mean_vol','p99_vol']}
        continue
    durs  = np.array([len(s)*5 for s in st])
    peaks = np.array([s.max() for s in st])
    vols  = np.array([s.sum() for s in st])
    storm_data[n] = dict(n_per_yr=len(st)/ny, mean_dur=durs.mean(),
                         p50_dur=np.median(durs), p95_dur=np.percentile(durs,95)/60,
                         max_dur=durs.max()/60,
                         mean_peak=peaks.mean(), p99_peak=np.percentile(peaks,99),
                         max_peak=peaks.max(),
                         mean_vol=vols.mean(), p99_vol=np.percentile(vols,99))
    s = storm_data[n]
    print(f"  {n:<10} N/yr={s['n_per_yr']:.1f}  dur_mean={s['mean_dur']:.0f}min  "
          f"dur_p50={s['p50_dur']:.0f}min  dur_p95={s['p95_dur']:.2f}hr  dur_max={s['max_dur']:.1f}hr  "
          f"peak_mean={s['mean_peak']:.4f}  peak_p99={s['p99_peak']:.4f}  peak_max={s['max_peak']:.4f}  "
          f"vol_mean={s['mean_vol']:.3f}mm  vol_p99={s['p99_vol']:.3f}mm")

# ── 6. Seasonality ───────────────────────────────────────────────────────
print(f"\n{sep}"); print("6. MONTHLY SEASONALITY"); print(sep)
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(f"  {'Version':<10} " + "  ".join(f"{m:>6}" for m in months))
for n in NAMES:
    df = DATA[n].copy(); ny = len(df)/(365*288)
    df['month'] = df['date'].dt.month
    mv = df.groupby('month')['avg_rainfall'].sum()/ny
    print(f"  {n:<10} " + "  ".join(f"{v:>6.1f}" for v in mv.values))

# ── 7. Divergence ────────────────────────────────────────────────────────
print(f"\n{sep}"); print("7. STATISTICAL DIVERGENCE"); print(sep)
real_h_nz = real_h[real_h > 0]
print(f"  {'Version':<10} {'KS_h':>8} {'KS_d':>8} {'Wass':>8} {'JSD':>8} {'ACF_RMSE':>10}")
for n in SYN:
    h = to_hourly(arr(n)); d = to_daily(arr(n)); h_nz = h[h>0]
    ks_h = stats.ks_2samp(real_h, h).statistic
    ks_d = stats.ks_2samp(real_d, d).statistic
    wass = stats.wasserstein_distance(real_h, h)
    if len(h_nz)>0:
        bins=np.linspace(0.001,max(real_h_nz.max(),h_nz.max()),150)
        hr,_=np.histogram(real_h_nz,bins=bins,density=True)
        hs,_=np.histogram(h_nz,bins=bins,density=True)
        hr+=1e-9;hr/=hr.sum();hs+=1e-9;hs/=hs.sum()
        jsd=float(jensenshannon(hr,hs))
    else: jsd=1.0
    acf_rmse = float(np.sqrt(np.mean((real_acf_h-compute_acf(h,24))**2)))
    print(f"  {n:<10} {ks_h:>8.4f} {ks_d:>8.4f} {wass:>8.5f} {jsd:>8.4f} {acf_rmse:>10.4f}")

# ── 8. Hourly extreme quantiles ──────────────────────────────────────────
print(f"\n{sep}"); print("8. EXTREME VALUE — Hourly Quantiles"); print(sep)
print(f"  {'Version':<10} {'P95_h':>8} {'P99_h':>8} {'P99.9_h':>10} {'Max_h':>8}")
for n in NAMES:
    h = to_hourly(arr(n))
    print(f"  {n:<10} {np.percentile(h,95):>8.4f} {np.percentile(h,99):>8.4f} "
          f"{np.percentile(h,99.9):>10.4f} {h.max():>8.4f}")

# ── 9. Multi-scale ───────────────────────────────────────────────────────
print(f"\n{sep}"); print("9. MULTI-SCALE AGGREGATION (CV & Skewness)"); print(sep)
print(f"  {'Version':<10} {'h_CV':>8} {'h_skew':>8} {'d_CV':>8} {'d_skew':>8} {'d_mean':>8}")
for n in NAMES:
    h = to_hourly(arr(n)); d = to_daily(arr(n))
    hcv = h.std()/h.mean() if h.mean()>0 else 0
    dcv = d.std()/d.mean() if d.mean()>0 else 0
    print(f"  {n:<10} {hcv:>8.3f} {stats.skew(h):>8.3f} {dcv:>8.3f} {stats.skew(d):>8.3f} {d.mean():>8.3f}")

print(f"\n{sep}")
print("DONE.")
