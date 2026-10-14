# Literature: coarse-then-fine generation of long 5-minute rainfall records

**Written:** 2026-10-15. **Question:** before building a two-resolution diffusion generator (coarse 30-min model
over 32 h, then a fine model that disaggregates to 5 min), has this, or the gap it addresses, been done before?
**Method:** 11 web searches (extended mode) on diffusion and deep-learning rainfall disaggregation, multi-resolution
and cascaded diffusion for time series, diffusion weather generators, generative rain for drainage, and long-range
dependence in generated series; abstracts read where reachable. Nature and Springer pages could not be opened
(login redirects), so those two entries rest on search summaries. Earlier reviews in this repository:
`docs/LITERATURE_RAINFALL_GENERATORS.md`, `docs/LITERATURE_CONTINUING_SEQUENCES.md`,
`research_notes/Rainfall generator literature review/classical_stochastic_generators.md`.

## Closest work found

| Work | What it does | Difference from the proposal |
| :--- | :--- | :--- |
| spateGAN-ERA5, npj Clim. Atmos. Sci. 2025 | conditional GAN: hourly 24 km ERA5 → 10 min, 2 km radar-like fields | gridded fields conditioned on reanalysis; not a generator of long continuous point records; GAN |
| LSTM sub-hourly disaggregation, SERRA 2025 | daily station rain → half-hourly; a normalisation layer conserves daily totals; states prior ML stopped at hourly | disaggregates observed daily data; 30 min; LSTM |
| Temporal disaggregation of spatial rainfall fields with GANs, HESS 25:3207, 2021 | daily → hourly radar fields, stochastic; reports training instability | gridded; hourly; GAN |
| Two-stage rainfall-forecasting diffusion, arXiv 2402.12779 | low-resolution forecast then super-resolution | forecasting, gridded |
| HourGlass, arXiv 2607.11457 (2026) | probabilistic 6-hourly → hourly downscaling of weather forecasts | forecasting, gridded |
| Diffusion precipitation downscaling (WassDiff arXiv 2410.00381; arXiv 2312.06071) | spatial (and spatio-temporal) super-resolution | spatial; not continuous records |
| mr-Diff, ICLR 2024 | coarse-to-fine multi-resolution diffusion | time-series forecasting |
| Cardiff (arXiv 2507.13366), MR-Traj (arXiv 2608.14570) | cascaded coarse-to-fine diffusion | movement trajectories |
| Cascaded diffusion (Ho et al. 2022), Imagen Video | low-resolution base model + super-resolution models, including in time | images and video |
| HyetosMinute (Bartlett–Lewis + adjusting), multiplicative cascades (Olsson 1998; Müller & Haberlandt 2018; Müller-Thomy 2020), WayDown, Q-CODA (2025), HESS 28:391 (2024) | classical "generate coarse, disaggregate fine" chains, down to 5 or 1 min | the established hydrology version, classical models; known weaknesses: autocorrelation too low, excess tiny intensities, structure across blocks |
| CTGAN synthetic rainfall for flood probability mapping, arXiv 2409.13936 | generative events for flood maps | events with KNN smoothing; not continuous 5-min records |
| Long-range dependence in financial markets, arXiv 2509.19663 | GANs fail to reproduce long memory; explicit long-memory mechanisms needed | finance; no pre-training horizon diagnostic |
| ImagenTime, arXiv 2410.19538 | diffusion on images of time series, short and long sequences | generic generation; not rainfall, no downstream horizon |

## Assessment

1. **Not found anywhere: a learned (diffusion or other deep) generator of long, continuous, sub-hourly point rainfall
   records for hydrological simulation.** Prior deep-learning rainfall work is forecasting, spatial downscaling, or
   disaggregation of *given* coarse data (observed or reanalysis), and stops at 10–30 min or hourly.
2. **The two-level structure itself is not new.** Coarse-then-fine is standard in hydrology (classical models) and
   in machine learning (cascaded diffusion; mr-Diff; trajectory cascades). Learned rainfall disaggregation also
   exists (LSTM to 30 min; GANs for gridded fields).
3. **Not found: a pre-training diagnostic of the horizon a downstream user needs.** Long-range-dependence failures of
   generative models are documented (finance), and forecasting papers note window limits, but nothing measures, before
   training, how much structure a block generator can carry at a site and checks it against a downstream simulator.
4. So a two-resolution diffusion rainfall generator would be **new as an application and combination** (diffusion at
   both levels, 5-min point records, horizon chosen by the diagnostic, judged by sewer loading), not as a method. The
   diagnostic is the most distinctive contribution.

## Implications for the design

* Conserve coarse totals exactly (as the LSTM paper's normalisation layer and microcanonical cascades do): the
  planned per-half-hour rescaling.
* Compare the fine level against a microcanonical cascade on the same coarse series: cascades are the standard tool
  for this step and their weaknesses (autocorrelation, tiny intensities) are where diffusion could win.
* Expect training instability or overfitting when stopping is chosen by loss alone (HESS 2021 GAN note).

## Before claiming novelty in writing

A Google Scholar / Web of Science check with the exact terms ("diffusion" or "denoising" with "rainfall
disaggregation", "temporal super-resolution", "sub-hourly", "stochastic rainfall generator", 2023–2026), and a
read of the spateGAN-ERA5 and LSTM papers in full.
