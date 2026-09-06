# What We Are Actually Optimizing

Reference note for the rainfall fork of ImagenFew. Written 2026-09-01, after auditing
`models/ImagenFew/ImagenFew.py`, `models/ImagenFew/handler.py` and
`regime_training/train_regime.py` against the v5–v7 results.

---

## 1. The objective

ImagenFew is an **EDM** model (Karras et al. 2022, *Elucidating the Design Space of
Diffusion-Based Generative Models*), not a DDPM. There is no `beta` schedule doing real
work and no `epsilon`-prediction. The model is trained as a **denoiser with an explicit
noise-level input**:

```
L(theta) = E_x E_sigma E_n [ lambda(sigma) * || D_theta(x + n ; sigma, c) - x ||^2 ]
```

with

| Symbol | Meaning | Where |
|---|---|---|
| `x` | one rainfall window, delay-embedded to an 8x8 image, StandardScaler-normalised | `RegimeDataset`, `DelayEmbedder` |
| `c` | one-hot seasonal-phase index (4 levels, ascending mean intensity) | `train_regime.py` |
| `n ~ N(0, sigma^2 I)` | the noise actually added | `ImagenFew.forward` |
| `sigma ~ LogNormal(P_mean=-1.2, P_std=1.2)` | noise level, sampled per **sample** | `ImagenFew.forward` |
| `lambda(sigma) = (sigma^2 + sigma_d^2) / (sigma * sigma_d)^2`, `sigma_d = 0.5` | EDM loss weighting | `ImagenFew.forward` |
| `D_theta` | preconditioned denoiser | `EDMPrecond` in `networks.py` |

The target is **`x` itself**, i.e. x-prediction. The preconditioning is

```
D_theta(y; sigma) = c_skip(sigma) * y + c_out(sigma) * F_theta( c_in(sigma) * y , c_noise(sigma), c )
c_skip = sigma_d^2 / (sigma^2 + sigma_d^2)
c_out  = sigma * sigma_d / sqrt(sigma^2 + sigma_d^2)
c_in   = 1 / sqrt(sigma^2 + sigma_d^2)
```

`lambda(sigma)` is chosen precisely so that `lambda * c_out^2 = 1`: after preconditioning,
the raw network `F_theta` sees a unit-variance target at every noise level. This is why the
loss curve is flat-ish and why its absolute value is not very interpretable — it is a
*uniformly weighted* regression across noise scales, by construction.

`sigma` has median `exp(-1.2) ~ 0.30` and spans roughly `[0.03, 3.0]` within one standard
deviation. Note that ImagenFew hardcodes `sigma_data = 0.5` while `RegimeDataset` uses
`StandardScaler`, which produces **std 1.0**. The preconditioning is therefore mis-tuned by
2x for this data. Not fatal — it shifts which noise levels get emphasis — but it is a free
improvement to correct.

### Why we no longer call this an HMM

Previous versions referred to conditioning variable `c` as an "HMM state" and the model as conditioned on Hidden Markov Model states. We have formally retired this terminology for three concrete mathematical and physical reasons (panel §2):
1. **The label is a periodic function of day-of-year identical in all 10 years:** The state assignment repeats identically across all 10 years on the calendar rather than identifying dynamically evolving meteorological weather states.
2. **Autocorrelation and violation of emission independence:** The HMM was fitted on a 14-day moving-average smoothed series. As a result, consecutive 5-minute samples have ~100% autocorrelation, completely voiding the core HMM conditional emission-independence assumption ($P(X_t \mid S_t, X_{<t}) = P(X_t \mid S_t)$).
3. **The transition matrix is a calendar lookup, not a Markov chain:** The 4×4 block transition matrix is estimated from a few hundred near-deterministic seasonal crossings with `smooth=1e-5` Laplace fill. Its off-diagonal entries merely capture "which season follows which" in the annual cycle rather than a stochastic 1st-order Markov process.

The conditioning variable $c$ is therefore accurately designated as a **one-hot seasonal-phase index (4 levels, ascending mean intensity)** under **seasonal-phase conditioning**.

### The extra spectral term

`train_regime.py` adds a second term the upstream handler does not have:

```
L = E [ lambda(sigma) * ( ||D - x||^2_time  +  ||FFT2(D) - FFT2(x)||^2 ) ]
```

See section 3 for why this term does almost nothing.

---

## 2. Why a plain L2 denoiser flattens rainfall extremes

At the optimum, the minimiser of the objective is the posterior mean
`D*(y; sigma) = E[x | y]`, which is the *exact* score function. So in the infinite-capacity,
infinite-data limit L2 is **not** biased — EDM sampling with an exact denoiser reproduces the
data distribution, tails and all. "L2 regresses to the mean" is folklore and is not by itself
the explanation.

The actual mechanism in this project is a **budget** problem, and it has three parts:

**(a) Rare events carry negligible expected loss.**
Astlingen at 5-min is 90.95% dry. After z-scoring, a dry step sits at `z ~ -0.145` and the
10-year maximum sits at `z ~ +119`. An extreme step contributes a huge *squared* residual
(order `10^4`) but occurs at a rate of order `10^-6`, so its share of the expected loss is
order `10^-2` — about 1%. Finite network capacity is allocated in proportion to expected
loss, so the tail of `E[x|y]` at low `sigma` is the first thing to be under-resolved.

**(b) Gradient clipping systematically attenuates exactly those events.**
`clip_grad_norm_(..., 1.0)`. A batch containing a cloudburst produces a gradient with a much
larger norm than a bulk dry batch, and clipping rescales it back to norm 1.0. The *direction*
survives, the *magnitude* does not. Bulk batches are typically under the threshold and pass
through unscaled. Net effect: heavy-rain updates are down-weighted relative to dry ones, on
top of already being rare. This is the most concrete and most fixable of the three.

**(c) EMA at decay 0.9999 has a ~10,000-step memory.**
Any correction learned from a rare batch is averaged away before it can accumulate, unless it
is reinforced often enough.

The empirical signature matches: v6 capped out at a 1.79 mm maximum against a real 5.56 mm,
and v5 at 4.13 mm, while both matched the *mean* wet intensity almost exactly. Getting the
bulk right and the tail wrong is the expected failure mode of (a)+(b)+(c) together.

### The fix now implemented

`regime_training/train_regime.py` applies a per-pixel weight to the time-domain term:

```
w(x) = 1 + alpha * relu(x)^gamma
```

- `relu` — because z-scoring puts dry steps at a small *negative* value, this leaves the 91%
  dry mass at exactly weight 1.0 and boosts only above-mean rainfall.
- `gamma = 0.5` — sub-linear on purpose. `z` reaches ~119 at the 10-year maximum, so a linear
  weight would let one cloudburst dominate an entire batch and re-create problem (b) in a new
  form. Measured profile at `alpha=0.3, gamma=0.5`: dry `1.00x`, P99 `1.54x`, P99.9 `2.04x`,
  10-yr max `4.27x`.
- weights are renormalised to batch-mean 1.0, so the loss magnitude, the usable learning rate
  and the grad-clip threshold stay comparable to the `alpha=0` baseline. This matters: without
  it, sweeping `alpha` silently sweeps the effective learning rate too.

`alpha = 0.0` is the default and reproduces v5/v6/v7 bit-for-bit.

`grad_clip` is now configurable, and `history.csv` logs `grad_norm` and `clip_frac` (the
fraction of steps that were actually clipped). **Check `clip_frac` before touching `alpha`** —
if it is high, raising the clip is the cheaper and more honest fix, because it removes a
distortion rather than adding a counter-distortion.

---

## 3. Two audit findings that affect how the results table should be read

### 3.1 The FFT term is ~1.5% of the loss and is not a spectral prior

`torch.fft.fft2(..., norm='forward')` divides by `N = H*W = 64`. By Parseval, for a residual
`r = D - x`:

```
sum_k |FFT2_fwd(r)_k|^2 = (1/N) * sum_n |r_n|^2
```

so the FFT term is **exactly 1/64 = 1.56% of the time term**. Verified numerically:
measured ratio `0.01560` against a predicted `0.015625`.

Worse, the mask is applied in the wrong domain. `fft_loss` is indexed by *frequency*, but the
code multiplies it by `(1 - x_img_mask)`, which is a *spatial* mask. What that actually does
is keep an arbitrary 3 of 8 frequency-column bins.

What the term does accomplish: with `seq_len=24, delay=8, embedding=8`, the delay embedding
fills only **3 of 8 columns** and the remaining 5 (62.5% of the image) are zero padding. The
time loss is masked off there, so nothing else constrains the denoiser's output in the padded
region — except the FFT term, which leaks across the whole image. A 5x error confined to the
padded region leaves the time term unchanged but multiplies the FFT term by 16x (measured).
So it is functioning as **padding housekeeping**, not as the frequency-fidelity prior it
looks like.

If you want genuine spectral pressure in v8: use `norm='ortho'` (Parseval-exact, ratio 1.0),
take the FFT of the **time series** over `seq_len` rather than of the zero-padded 2-D image,
and drop the spatial mask. A `fft_weight` knob now exists to scale the term deliberately;
it defaults to 1.0, i.e. current behaviour.

### 3.2 Two code paths, two different losses

| Path | Used by | Loss |
|---|---|---|
| `models/ImagenFew/handler.py:44` (via `run.py`) | **v3, v4** | time term only — **no FFT term** |
| `regime_training/train_regime.py` | **v5, v6, v7** | time + FFT term |

`ImagenFew.loss_fn` (which does include the FFT term) is dead code — nothing calls it.

Given 3.1, this discrepancy shifts the loss by about 1.5% and is not what distinguishes v5–v7
from v3/v4. But the results table should say which objective produced which checkpoint rather
than implying one setup throughout.

---

## 4. Change log

| Date | Change | Default |
|---|---|---|
| 2026-09-01 | `intensity_weight_alpha/gamma/normalize` — heavy-tail reweighting of the time term | `alpha=0.0` (off, reproduces v5–v7) |
| 2026-09-01 | `grad_clip` configurable; `grad_norm` + `clip_frac` logged to `history.csv` | `1.0` (unchanged) |
| 2026-09-01 | `fft_weight` scale on the spectral term | `1.0` (unchanged) |
| 2026-09-01 | `regime_*_p999` logged per epoch + `tail_health.png` | always on |

### Suggested v8 sweep, in order of expected value per GPU-hour

1. **Diagnose first, one run.** Train with defaults and read `clip_frac`. If it is above ~20%,
   raise `grad_clip` to 5.0 and re-run before anything else.
2. **`seq_len` 24 -> 288.** See `ACCEPTANCE_CRITERIA.md` — this is very likely the larger
   problem, and it is not a loss issue at all.
3. **`alpha` in {0.3, 1.0}** at `gamma=0.5`, tail health read off `tail_health.png`.
4. `sigma_data` 0.5 -> 1.0 to match the StandardScaler output.
