# Deep generative models for synthetic rainfall / precipitation (2019-2026)

Scope note: about 20 web searches and fetches, run 2026-10-01. Items marked **[verified]** were confirmed through a primary page (journal, arXiv abstract, or publisher) during this session. Items marked **[from search snippet]** were confirmed only through search-result text. Items marked **[uncertain]** rest on prior knowledge that was not re-checked here. Compared with our project (EDM diffusion, 5-min point rainfall, 10-year continuous records, independent 64-step blocks, single site, used as RL training input), the closest prior work is thin. **No deep generative paper found in this session produces multi-year continuous 5-min point rainfall for urban drainage control.**

## 1. Which deep generative models generate rainfall time series or space-time fields? (catalogue)

### Takeaway
Most deep generative precipitation work is **gridded, conditional, and short-horizon**: downscaling (GAN, then diffusion from about 2023) and nowcasting. True unconditional or weakly conditional *stochastic generators* of long synthetic records are few: Ji et al. 2024 (GAN, hourly, 14 stations, 30 years), Guevara et al. 2021 (GAN/VAE vs classical weather generators, multisite daily), Puchko et al. DeepClimGAN, Besombes et al. 2021, DiffESM (monthly-to-daily diffusion), the EVT-GAN family for extremes, and cBottle (a global km-scale diffusion climate generator).

### Cited Findings

**A. Stochastic weather/rainfall GENERATORS (synthetic records), closest to our use case**

- **Ji, Mirzaei, Lai, Dehghani et al. (2024), "Implementing generative adversarial network (GAN) as a data-driven multi-site stochastic weather generator for flood frequency estimation", *Environmental Modelling & Software* 172, 105896.** A GAN synthesizes **hourly precipitation over 30 years at 14 stations**. The output drives an hourly-calibrated SWAT model for streamflow and flood frequency. The authors claim more authentic spatial correlation of extreme rainfall events than conventional generators. [from search snippet; ScienceDirect blocked fetch] — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815223002827)
- **Guevara, Borges, Watson, Zadrozny (2021), "A comparative study of stochastic and deep generative models for multisite precipitation synthesis", ICML 2021 Workshop Tackling Climate Change with ML.** Compares the classical generators IBMWeathergen and RGeneratePrec with a GAN and a VAE for multisite precipitation. The authors call the results "preliminary" and note that "there only a few works comparing and evaluating promising deep learning models for weather generation against classical approaches." [verified] — [arXiv 2107.08074](https://arxiv.org/abs/2107.08074)
- **Puchko et al. (2020), "DeepClimGAN: A High-Resolution Climate Data Generator".** A GAN that generates spatio-temporal daily climate fields (precipitation and temperature) as an emulator of Earth system model output. [title/arXiv verified via search; detailed design (conditioning on monthly context, 32-day output) **uncertain**] — [arXiv 2011.11705](https://arxiv.org/pdf/2011.11705)
- **Besombes, Pannekoucke, Lapeyre, Sanderson, Thual (2021), "Producing realistic climate data with generative adversarial networks", *Nonlinear Processes in Geophysics* 28, 347-370.** A Wasserstein GAN trained on the PLASIM simplified GCM climate. It produces realistic multivariate weather states, including geostrophic balance. These are single snapshots, not time series. [from search snippet] — [NPG](https://npg.copernicus.org/articles/28/347/2021/)
- **Bassetti, Hutchinson, Tebaldi, Kravitz (2024), "DiffESM: Conditional Emulation of Temperature and Precipitation in Earth System Models With 3D Diffusion Models", *JAMES*.** A 3D (space and time) diffusion model that generates **daily** precipitation or temperature sequences from **monthly mean** inputs. It matches ESM output on extreme-relevant metrics such as dry spells and precipitation intensity on extremely wet days. This is temporal disaggregation used as a generator, and the output is month-long chunks. [from search snippet] — [JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004194); [arXiv 2409.11601](https://arxiv.org/pdf/2409.11601); earlier version [arXiv 2304.11699](https://arxiv.org/pdf/2304.11699); joint temperature-precipitation follow-up [arXiv 2404.08797](https://arxiv.org/pdf/2404.08797)
- **NVIDIA, "Climate in a Bottle (cBottle)" (2025), arXiv 2505.06474.** A two-stage diffusion model on a HEALPix grid: a global 100 km generator followed by a 16x patch-based super-resolution stage to 5 km. Conditioning is on time of day, day of year and SST. It samples atmospheric states directly from the distribution without autoregressive rollout. It is trained on km-scale simulations plus about 50 years of observations. [from search snippet] — [arXiv 2505.06474](https://arxiv.org/abs/2505.06474)
- **Klemmer et al. (2021), "Generative modeling of spatio-temporal weather patterns with extreme event conditioning".** A spatio-temporal GAN whose generation can be conditioned on extreme events. [title/arXiv from search; venue (likely an ICLR 2021 workshop) and authors **uncertain**] — [arXiv 2104.12469](https://arxiv.org/pdf/2104.12469)
- **"A spatial weather generator based on conditional deep convolution GAN (cDCGAN)".** Generates high-resolution weather from GCM outputs for four areas in China under four SSP scenarios. [from search snippet; authors/year not confirmed] — [ResearchGate](https://www.researchgate.net/publication/374475227_A_spatial_weather_generator_based_on_conditional_deep_convolution_generative_adversarial_nets_cDCGAN)
- **Rampal et al. (2025), "A Reliable Generative Adversarial Network Approach for Climate Downscaling and Weather Generation", *JAMES*.** A conditional GAN that is framed as both a downscaler and a stochastic weather generator. [from search snippet] — [JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004668)

**B. Extreme-focused generative models (EVT + deep generative)**

- **Boulaguiem, Zscheischler, Vignotto, van der Wiel, Engelke (2022), "Modeling and simulating spatial extremes by combining extreme value theory with generative adversarial networks" (evtGAN), *Environmental Data Science* (Cambridge).** EVT marginals are combined with a GAN for spatial dependence. The model is applied to summer temperature maxima and winter precipitation maxima over western Europe. It beats classical GANs and standard statistical approaches, but **precipitation-extreme dependence is captured worse than temperature** because precipitation fields are less spatially coherent. [from search snippet; author list partly uncertain] — [Cambridge EDS](https://www.cambridge.org/core/journals/environmental-data-science/article/modeling-and-simulating-spatial-extremes-by-combining-extreme-value-theory-with-generative-adversarial-networks/AA3CEBA3CF1EF3C9C8D182C108289D7F); [arXiv 2111.00267](https://arxiv.org/pdf/2111.00267)
- **Bhatia, Jain, Hooi (2021), "ExGAN: Adversarial Generation of Extreme Samples", AAAI 35(8), 6750-6758.** Uses EVT to let the user set the desired extremeness and its probability. Demonstrated on **daily precipitation grids over the US**, where it produces extreme rainfall samples resembling real floods. The authors note that plain GANs generate *typical* rather than extreme samples. [from search snippet] — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/16834); [arXiv 2009.08454](https://arxiv.org/pdf/2009.08454)
- **"Combining deep generative models with extreme value theory for synthetic hazard simulation: a multivariate and spatially coherent approach" (2023)** and **"Simulating spatial multi-hazards with generative deep learning", *NHESS* 26 (2026).** Both pipelines fit EVT to footprints extracted from reanalysis and then train a GAN on the transformed footprints, covering wind, pressure and rainfall. [from search snippet] — [arXiv 2311.18521](https://arxiv.org/html/2311.18521); [NHESS](https://nhess.copernicus.org/articles/26/1663/2026/)
- **"ExceedGAN: simulation above extreme thresholds using GANs" (2026, *Extremes*)** and **"Simulation of multivariate extremes: A Wasserstein-Aitchison GAN approach" (2026, *Extremes*).** Statistical-extremes GANs, both 2026. [from search snippet; precipitation application not confirmed] — [Springer 1](https://link.springer.com/article/10.1007/s10687-026-00528-9); [Springer 2](https://link.springer.com/article/10.1007/s10687-026-00530-1)

**C. Downscaling and temporal disaggregation (conditional; context only)**

- **Scher & Peßenteiner (2021), "Technical Note: Temporal disaggregation of spatial rainfall fields with generative adversarial networks", *HESS* 25, 3207.** A GAN maps daily-sum fields (16x16 grid) to **hourly** 24x16x16 sequences. It is trained on SMHI 5-min radar data for 2009-2018, aggregated to hourly. Weaknesses reported: it **cannot produce exact zeros**, it generates too many ~1 mm/h hours, it underestimates the diurnal cycle, adding day-of-year/longitude inputs made it *worse*, choosing when to stop training needs expert judgment, and it degrades on larger domains. [verified] — [HESS](https://hess.copernicus.org/articles/25/3207/2021/)
- **Glawion, Polz, Kunstmann, Fersch, Chwala (2023), "spateGAN: Spatio-Temporal Downscaling of Rainfall Fields Using a cGAN Approach", *Earth and Space Science*.** A video-super-resolution cGAN trained on **10 years of German radar**. It increases spatial resolution 16x and temporal resolution 6x. [from search snippet] — [ESS](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA002906)
- **Glawion et al. (2025), "Global spatio-temporal ERA5 precipitation downscaling to km and sub-hourly scale using generative AI" (spateGAN-ERA5), *npj Climate and Atmospheric Science* 8, 219.** Downscales ERA5 from 24 km / 1 h to **2 km / 10 min**. It is trained only on Germany and validated in the US and Australia, and it produces ensembles. **This is the most relevant sub-hourly deep generative work found**, though it is gridded and conditional on ERA5. [verified] — [arXiv 2411.16098](https://arxiv.org/abs/2411.16098); [npj](https://www.nature.com/articles/s41612-025-01103-y)
- **Harris, McRae, Chantry, Dueben, Palmer (2022), "A Generative Deep Learning Approach to Stochastic Downscaling of Precipitation Forecasts", *JAMES*.** Compares a cGAN with a VAE-GAN. The VAE-GAN gives "much sharper and better-calibrated" results at 10x resolution. [from search snippet; author list from prior knowledge, **uncertain**] — [JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003120); [arXiv 2204.02028](https://arxiv.org/abs/2204.02028)
- **Addison et al. (2026), "Machine Learning Emulation of Precipitation From km-Scale UK Regional Climate Simulations Using a Diffusion Model" (CPMGEM), *JAMES* 18(3).** A score-based diffusion model that downscales 60 km GCM inputs to 8.8 km **daily** precipitation, emulating a 2.2 km convection-permitting model. It reproduces the heaviest events and the 21st-century change signal. [from search snippet] — [JAMES](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005140); [arXiv 2407.14158](https://arxiv.org/pdf/2407.14158); code [github.com/henryaddison/mlde](https://github.com/henryaddison/mlde)
- **Mardani et al. (2025), "Residual corrective diffusion modeling for km-scale atmospheric downscaling" (CorrDiff), *Communications Earth & Environment*.** A UNet regression followed by a diffusion residual, 25 km to 2 km over Taiwan, including radar reflectivity and precipitation proxies. [**uncertain**, not re-verified this session]
- **RainScaleGAN (2025), *AI for the Earth Systems* 4(3).** A conditional GAN for rainfall downscaling. [from search snippet] — [AMS AIES](https://journals.ametsoc.org/view/journals/aies/4/3/AIES-D-24-0074.1.xml)
- **WassDiff (2024), "Generative Precipitation Downscaling using Score-based Diffusion with Wasserstein Regularization".** Adds a Wasserstein distribution-matching term to the score loss. [from search snippet] — [arXiv 2410.00381](https://arxiv.org/html/2410.00381v1)
- **Intercomparisons of generative downscalers.** One is a GMD 2026 paper comparing U-Net, cWGAN and conditional DDPM: "U-Net systematically underestimates the upper quantiles ... >20 mm/day", while WGAN and DDPM give heavier tails. Another is an arXiv 2025 intercomparison of generative methods at fine scales. [from search snippet] — [GMD 19, 7545 (2026)](https://gmd.copernicus.org/articles/19/7545/2026/); [arXiv 2512.13987](https://arxiv.org/pdf/2512.13987)
- **Other diffusion downscalers (2024-2026), listed only.** Conditional wavelet diffusion ([arXiv 2507.01354](https://arxiv.org/pdf/2507.01354)), foundation-model-conditioned diffusion ([arXiv 2608.25858](https://arxiv.org/pdf/2608.25858)), flow matching for convective-scale downscaling ([arXiv 2606.00281](https://arxiv.org/pdf/2606.00281)), and a likelihood-based spatially consistent generator ([arXiv 2407.04724](https://arxiv.org/pdf/2407.04724)). [from search snippet] Spatiotemporal video diffusion downscaling (Srivastava et al., NeurIPS 2024) is [**uncertain**, not re-verified].

**D. Nowcasting / forecasting (context only)**

- **DGMR**: Ravuri et al. (2021), "Skilful precipitation nowcasting using deep generative models of radar", *Nature* 597. A GAN that produces 90-min nowcasts from UK radar. [prior knowledge; **uncertain** on exact details] It is described as state of the art in generative nowcasting, with a hinge loss plus a regularizer that matches true precipitation amounts. — [LDCast paper cites it](https://arxiv.org/pdf/2304.12891)
- **LDCast**: Leinonen et al. (2023), "Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification". [title verified via search] — [arXiv 2304.12891](https://arxiv.org/pdf/2304.12891)
- **Other diffusion nowcasters.** Asperti et al. "Precipitation nowcasting with generative diffusion models" (*Applied Intelligence*, 2025; [arXiv 2308.06733](https://arxiv.org/pdf/2308.06733)), RNDiff ([arXiv 2402.13737](https://arxiv.org/pdf/2402.13737)), a two-stage rainfall-forecasting diffusion model ([arXiv 2402.12779](https://arxiv.org/pdf/2402.12779)), a transformer generative extreme nowcaster ([arXiv 2403.03929](https://arxiv.org/pdf/2403.03929)), and BlockGPT, a frame-level autoregressive rainfall model ([arXiv 2510.06293](https://arxiv.org/pdf/2510.06293)). PreDiff (Gao et al., NeurIPS 2023), DiffCast (CVPR 2024), CasCast (ICML 2024) and GenCast (Price et al., *Nature* 2025) are [**uncertain**, not re-verified]. Asperti et al. author attribution is also [**uncertain**].
- **Station-level zero-inflated forecasting.** ZIDF, "From Noise to Precision: A Diffusion-Driven Approach to Zero-Inflated Precipitation Prediction" (2025), adds Gaussian noise to smooth the zero-inflated target, forecasts with a Transformer, then denoises with diffusion. It reports up to 56.7% lower MSE than a Non-stationary Transformer. MZ-Rain (2026) is a zero-inflated station nowcaster. [from search snippet] — [arXiv 2509.10501](https://arxiv.org/abs/2509.10501); [arXiv 2609.04864](https://arxiv.org/pdf/2609.04864)

### Inferences
- A rough taxonomy by count of what turned up in this session: downscaling (largest group), nowcasting, extremes/EVT-GANs, then a small group of true synthetic-record generators. Only Ji et al. 2024 combines a deep generative model, a multi-year continuous output, and a hydrological impact model in the way our project does.
- The architecture trend runs GAN (2019-2022), then GAN + EVT and VAE-GAN (2021-2023), then diffusion (2023 onward). Our EDM choice matches the current state of the art for distribution fidelity.

### Gaps
- No full text was obtained for Ji et al. 2024 (paywalled). Its architecture (recurrent vs convolutional), how it handles zeros, and how 30-year sequences are assembled (chunking) are unknown.
- Training details for Guevara et al. 2021 (dataset, temporal resolution) were not retrieved beyond the abstract.
- CorrDiff, GenCast, PreDiff, DiffCast, CasCast and the video-diffusion downscaler were not re-verified in this session.

## 2. Generators vs downscaling vs nowcasting; what is generated and how long sequences are produced

### Takeaway
Almost every model generates **fixed-size chunks**: one snapshot (Besombes, evtGAN, ExGAN, cBottle), one day of hours (Scher & Peßenteiner), one month of days (DiffESM), or a short radar clip (DGMR, LDCast, spateGAN). Long records come from conditioning on an external coarse driver (ERA5, GCM monthly means) or from autoregressive rollout. No paper found builds a multi-year sub-hourly point record out of independent chunks the way we do.

### Cited Findings
- DiffESM produces daily sequences consistent with **monthly** means. Long-term coherence is inherited from the monthly driver, not learned. — [JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004194)
- cBottle samples states "directly from the full distribution of atmospheric states, avoiding auto-regressive rollout". It is conditioned on time of day, day of year and SST. — [arXiv 2505.06474](https://arxiv.org/abs/2505.06474)
- Scher & Peßenteiner generate 24 hourly steps conditioned on a daily sum. The day boundaries are therefore set by the conditioning. — [HESS](https://hess.copernicus.org/articles/25/3207/2021/)
- spateGAN-ERA5 produces 10-min / 2 km fields conditioned on hourly ERA5. In principle it can make arbitrarily long records wherever ERA5 exists, but the record is not *synthetic weather*: its storm timing is fixed by ERA5. — [npj](https://www.nature.com/articles/s41612-025-01103-y)
- Ji et al. produce a 30-year hourly multisite record. How the chunks were assembled is not known (see Gaps). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815223002827)
- **Generic long-sequence time-series diffusion (methods relevant to our block-independence problem):**
  - **WaveStitch (2025)** generates segments in parallel and keeps them coherent with an overlap-alignment "stitching" mechanism that refines them iteratively. It reports 166x speed-up over autoregressive generation with better quality. — [arXiv 2503.06231](https://arxiv.org/html/2503.06231)
  - **TransFusion** (diffusion + transformer) generates long time series up to length 384. — [arXiv 2307.12667](https://arxiv.org/pdf/2307.12667v2); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666827025000350)
  - **Stage-Diff (2025)** does stage-wise long-term time-series generation with diffusion. — [arXiv 2508.21330](https://arxiv.org/pdf/2508.21330)
  - **CompDiffuser** ("Generative Trajectory Stitching through Diffusion Composition", 2025) models trajectories as overlapping chunks with one bidirectional diffusion model conditioned on neighbouring chunks. — [arXiv 2503.05153](https://arxiv.org/pdf/2503.05153)
  - **TIMED (2025)** refines diffusion time-series generation with adversarial and autoregressive components. — [arXiv 2509.19638](https://arxiv.org/pdf/2509.19638)
- BlockGPT (2025) models rainfall frame by frame autoregressively: a transformer over frame blocks for spatio-temporal rainfall. — [arXiv 2510.06293](https://arxiv.org/pdf/2510.06293)

### Inferences
- Our E2 context-chained generation (conditioning each block on the previous block's tail) is the same family as autoregressive diffusion, CompDiffuser and WaveStitch's overlap conditioning. WaveStitch and CompDiffuser are the most direct method references for fixing storms that are cut at block boundaries.
- A hierarchical alternative supported by the literature is the DiffESM/Scher pattern: generate a coarse sequence (daily or hourly totals or an HMM state sequence) and disaggregate it conditionally. Our HMM-state conditioning is a weak version of this. Conditioning on a coarse *amount* (for example 6-hourly totals from a cheaper generator) would carry multi-block storm persistence.

### Gaps
- No paper found evaluates storm-duration distributions across chunk boundaries for a chunked rainfall generator.

## 3. Reported strengths and weaknesses: intermittency, extremes, persistence, spatial coherence, seasonality

### Takeaway
**Intermittency (exact zeros) and extremes are the recurring weaknesses.** GANs cannot output exact zeros without a threshold and over-produce light rain. Plain GANs produce typical rather than extreme samples, which motivated the EVT hybrids. Diffusion and GANs beat deterministic U-Nets on tails, but heavy-tail behaviour remains an open methods problem. Persistence and long-storm structure are rarely evaluated. Seasonal conditioning sometimes *hurts* (Scher & Peßenteiner).

### Cited Findings
- **Zeros.** The Scher & Peßenteiner GAN "cannot produce exactly zero precipitation values" and generates too many ~1 mm/h events. — [HESS](https://hess.copernicus.org/articles/25/3207/2021/)
- **Zeros in forecasting.** ZIDF smooths the zero-inflated target with Gaussian noise so that gradient-based training is not overwhelmed by zeros. — [arXiv 2509.10501](https://arxiv.org/abs/2509.10501)
- **Extremes.** "existing GAN-based approaches ... generate typical rather than extreme samples" (motivation for ExGAN). — [arXiv 2009.08454](https://arxiv.org/pdf/2009.08454)
- **Extremes, spatial.** evtGAN captures dependence in precipitation extremes worse than in temperature extremes because of lower spatial coherence. — [Cambridge EDS](https://www.cambridge.org/core/journals/environmental-data-science/article/modeling-and-simulating-spatial-extremes-by-combining-extreme-value-theory-with-generative-adversarial-networks/AA3CEBA3CF1EF3C9C8D182C108289D7F)
- **Generative vs deterministic tails.** U-Net underestimates quantiles above 20 mm/day, while WGAN and DDPM come closer to the 1:1 line. — [GMD 2026](https://gmd.copernicus.org/articles/19/7545/2026/)
- **Heavy-tailed diffusion.** Pandey et al., "Heavy-Tailed Diffusion Models" (ICLR 2025), replace the Gaussian noise prior with a Student-t prior for heavy-tailed data. [from search snippet; authors **uncertain**] Some precipitation downscaling work adopts the Student-t variant for log-precipitation residuals. — [arXiv 2410.14171](https://arxiv.org/pdf/2410.14171); [arXiv 2512.13987](https://arxiv.org/pdf/2512.13987)
- **Heavy-tailed flow matching.** "Tail Annealing for Heavy-Tailed Flow Matching" (2026) argues that a parameter-free soft-log transform sign(x)·log(1+|x|) turns Pareto tails into exponential-style tails, so standard flow matching works without architectural changes. — [arXiv 2605.20068](https://arxiv.org/pdf/2605.20068)
- **Daily extremes in DiffESM.** It matches ESM extreme metrics: dry spells and intensity of extremely wet days. — [JAMES](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004194)
- **Seasonality conditioning.** Adding day-of-year and longitude inputs degraded the Scher & Peßenteiner GAN. — [HESS](https://hess.copernicus.org/articles/25/3207/2021/)
- **Spatial coherence.** Ji et al. claim the GAN reproduces spatial correlation of extreme rainfall better than conventional multisite generators. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815223002827)
- **Training stability.** GAN stopping needs expert judgment. — [HESS](https://hess.copernicus.org/articles/25/3207/2021/)

### Inferences
- Our diffusion model avoids GAN mode-collapse and stopping-point issues, but the exact-zero problem still applies to any continuous generator. A threshold or an explicit occurrence channel is the standard workaround (inferred; the occurrence-channel idea is not sourced here).
- Our seq_len-64 limit on storm duration is a "persistence" failure. The deep generative literature reports this failure mode rarely, because most models are conditioned on a coarse driver that supplies persistence.

### Gaps
- No deep generative paper found reports wet-spell or storm-duration distributions at sub-hourly resolution.

## 4. Sub-hourly (5-10 min) work, multi-year continuous records, urban drainage / sewer / RL use

### Takeaway
Sub-hourly deep generative rainfall exists only as **gridded conditional downscaling** (spateGAN, spateGAN-ERA5 at 10 min) or as 5-min radar training data aggregated to hourly (Scher & Peßenteiner). The only sewer-related deep generative work found is Bakhshipour et al.'s Bayesian GAN for combined-sewer flow data augmentation. No paper found uses deep generative rainfall to train **RL** sewer or valve controllers, so our project appears novel on that axis.

### Cited Findings
- spateGAN-ERA5: 10-min, 2 km output from 1 h, 24 km ERA5, trained on Germany. — [arXiv 2411.16098](https://arxiv.org/abs/2411.16098)
- spateGAN: 6x temporal and 16x spatial super-resolution, trained on 10 years of German radar. — [ESS](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA002906)
- **Bakhshipour, Koochali, Dittmer, Haghighi, Ahmad, Dengel (2022/2023), "A Bayesian Generative Adversarial Network (GAN) to Generate Synthetic Time-Series Data, Application in Combined Sewer Flow Prediction", WDSA/CCWI 2022.** A GAN generates synthetic precipitation and storage-tank inflow series for a **small town in Germany** to augment data-driven combined-sewer flow prediction. Augmentation improved **peak-flow** prediction, while the non-augmented model was better in dry weather, so the authors suggest an ensemble. — [arXiv 2301.13733](https://arxiv.org/pdf/2301.13733)
- Ji et al. 2024: a 30-year hourly multisite GAN record feeding SWAT for flood frequency, an impact-model use analogous to ours. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815223002827)
- A ScienceDirect 2026 paper, "High-resolution flood probability mapping using generative machine learning with large-scale synthetic precipitation and inundation data", uses CTGAN to conditionally generate synthetic precipitation point clouds. [from search snippet] — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1093968726017688)
- For contrast, classical (non-deep) generators already reach 5-min to decade scales. One example is a randomized Bartlett-Lewis model that reproduces properties "across the timescales from several minutes to a decade" (*J. Hydrology*, 2020). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169420306107)

### Inferences
- The relevant baselines for our thesis are classical sub-hourly generators (Bartlett-Lewis, STORAGE, MRC cascades) and not deep generative models, because no deep generative point generator at 5 min was found.
- Bakhshipour et al. is the nearest application-domain citation. It is a German combined-sewer case, though prediction-oriented rather than RL.

### Gaps
- No deep generative rainfall model used for RL training was found. A dedicated search on "reinforcement learning" + "synthetic rainfall" + "generative" was not run within the tool budget.

## 5. Time-series diffusion on zero-inflated data, chunked long-sequence generation, and log/asinh transforms

### Takeaway
Transforms: log(1+x) followed by normalization is the de facto choice for precipitation in diffusion and GAN models. Soft-log / signed-log transforms and Student-t noise are the current answers to heavy tails. **No precipitation paper using asinh was found.** Zero inflation in diffusion is handled by noise-smoothing (ZIDF) or ignored. Chunk-coherence methods (WaveStitch, CompDiffuser, TransFusion, Stage-Diff) exist in generic time-series ML but have not been applied to rainfall.

### Cited Findings
- A DDPM precipitation downscaler applies "log(1+x) transformation followed by min-max normalization" for numerical stability; the same intercomparison notes heavy-tailed log residuals. — [arXiv 2512.13987](https://arxiv.org/pdf/2512.13987)
- The soft-log sign(x)·log(1+|x|) "compress[es] heavy Pareto-style tails into light exponential-style tails". — [arXiv 2605.20068](https://arxiv.org/pdf/2605.20068)
- Student-t heavy-tailed diffusion. — [arXiv 2410.14171](https://arxiv.org/pdf/2410.14171)
- ZIDF Gaussian-perturbation smoothing of zero-inflated precipitation. — [arXiv 2509.10501](https://arxiv.org/abs/2509.10501)
- WaveStitch, CompDiffuser, TransFusion, Stage-Diff, TIMED (see Section 2). — [WaveStitch](https://arxiv.org/html/2503.06231); [CompDiffuser](https://arxiv.org/pdf/2503.05153); [TransFusion](https://arxiv.org/pdf/2307.12667v2); [Stage-Diff](https://arxiv.org/pdf/2508.21330); [TIMED](https://arxiv.org/pdf/2509.19638)
- WassDiff adds a Wasserstein distribution term to counter distribution mismatch in diffusion precipitation output. — [arXiv 2410.00381](https://arxiv.org/html/2410.00381v1)

### Inferences
- Our v13 (log1p) and v14 (asinh, s = 0.035) transforms match literature practice. asinh behaves like the soft-log (linear near zero, logarithmic in the tail), so the "Tail Annealing" paper is the closest theoretical justification to cite. Our ablation result (both reached 14/18, beating StandardScaler) is consistent with that paper's claim.
- Student-t noise (heavy-tailed diffusion) and a Wasserstein or tail-aware loss term (WassDiff) are literature-backed next options for our under-generated extremes.

### Gaps
- No precipitation-specific asinh paper was found.
- No paper found directly evaluates diffusion time-series models on zero-inflated *sub-hourly* rainfall generation (as opposed to forecasting).
