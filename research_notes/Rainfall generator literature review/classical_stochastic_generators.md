# Classical (non-deep-learning) stochastic rainfall generators for sub-daily / sub-hourly point rainfall

Scope: families of classical generators that can produce long, continuous, 5-10 min point rainfall
for urban hydrology, their state around 2015-2026, and how they are evaluated. Written to compare
against our diffusion generator (10-year continuous 5-min Astlingen series for RL valve control) and
our own Gaussian AR / latent Gaussian AR baselines (Gaussian-copula models in the literature's terms) (the first fits fragment rain: 75% single-step
showers, no IDF cells in band; refitted on 2026-10-05 so the latent correlation matches rain
occurrence, the latent Gaussian AR gets showers and long storms right but over-produces 1-24 h extremes,
1.3-1.9x real).

Verification note: every bullet below has a link I opened or saw in search results during this
session. Where a fact comes from a secondary mention (for example a review's reference list) rather
than the primary paper, the bullet says so. Several foundational papers I know of but did not open
are listed under Gaps, not presented as findings.

---

## 1. Poisson-cluster (Neyman-Scott / Bartlett-Lewis rectangular pulse) models

### Takeaway
Poisson-cluster rectangular-pulse models (NSRP, BLRP and their randomised variants) are the
best-established mechanistic generators. They are continuous-time and fitted by the generalised
method of moments (GMM) to aggregated statistics. Their known weakness is under-estimating hourly and
sub-hourly extremes. Since 2014 a line of work (Kaczmarska, Onof & Wang, Cross, Park, Kim)
has largely closed that gap at 5 min, using German 5-min data (Bochum) as the benchmark. This is the
strongest classical competitor to our model for a single-site 5-min series.

### Cited Findings
**Structure and fitting**
- Storms arrive as a Poisson process. Each storm spawns rain cells (rectangular pulses with random
  intensity X and duration D), and the intensity at time t is the sum of all active cells. Cells can
  overlap, and a storm can contain zero-intensity gaps. In Neyman-Scott the cell displacements from the
  storm origin are i.i.d. In Bartlett-Lewis the cell arrivals form a Poisson process that ends after a
  random storm duration. — [Northrop 2024, Annu. Rev. Stat. Appl. 11:51-74 (UCL preprint)](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf); [journal DOI](https://doi.org/10.1146/annurev-statistics-040622-023838)
- Because the models are defined in continuous time, one fit can in principle be used at any
  aggregation level. They are typically fitted to totals over discrete intervals by GMM: parameters
  minimise a weighted distance between analytical and observed statistics such as mean, variance,
  lag-1 autocorrelation, skewness and proportion dry at several scales. They are usually fitted
  separately per calendar month to handle seasonality. — [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)
- Northrop warns that using such models to extrapolate to 5-min totals for urban drainage "may not
  be warranted". He says good sub-hourly data for fitting or validation are better, and that the
  rectangular-pulse assumption may need refining for realistic short-time-scale behaviour. —
  [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)
- Other inference routes: approximate Bayesian computation (Aryal & Jones 2020, 2021), Whittle or
  composite-likelihood estimating functions (Jesus & Chandler 2011, 2017), and local GMM for
  non-stationarity (Kaczmarska, Isham & Northrop 2015, *Environmetrics* 26:312-325). The GMM
  objective may have local minima, and log-reparameterisation and good starting values matter.
  Software: MOMFIT (Chandler 2016) fits single-site models, and RainSim (Burton et al. 2008, *Env.
  Modelling & Software* 23:1356-1369) simulates spatial-temporal NSRP. — [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf); [Kaczmarska et al. 2015](https://onlinelibrary.wiley.com/doi/full/10.1002/env.2338)
- Adding the coefficient of skewness to the fitting set, with a heavier cell-intensity distribution
  (gamma or Weibull), improves extremes (Cowpertwait 1998). Cameron et al. (2000) used a generalised
  Pareto distribution for large cell intensities. Verhoest et al. (2010, *Hydrol. Processes*
  24:3439-3445) asked whether point models preserve extreme flood statistics. — [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)

**Known weakness: sub-hourly extremes**
- "BL models tend to underestimate hourly and sub-hourly extremes while overestimating daily
  extremes." — [Onof & Wang 2020, HESS 24:2791-2815](https://hess.copernicus.org/articles/24/2791/2020/) (summary via search snippet; same point in [Park, Onof & Kim 2019, HESS 23:989-1014](https://hess.copernicus.org/articles/23/989/2019/))

**Fine-scale developments (2007-2025)**
- Cowpertwait, Isham & Onof (2007, *Proc. R. Soc. A* 463:2569-2587) represent cells as a random
  number of instantaneous bursts to obtain fine-scale structure. — [Northrop 2024 ref list](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)
- Kaczmarska, Isham & Onof (2014, *Hydrol. Sci. J.* 59(11):1972-1991), "Point process models for
  fine-resolution rainfall": a BL model for sub-hourly use. It replaces rectangular cells with finite
  Poisson processes of instantaneous pulses and adds a variant with dependence between cell
  intensity and duration. It systematically compares variants fitted to **5-min data from Germany**. —
  [arXiv preprint](https://arxiv.org/pdf/1309.4632)
- Onof & Wang (2020, HESS): previous papers constrained the RBL shape parameter α too tightly
  (RBL1: α > 2, not α > 4; RBL2: α > 0, not α > 1). With the wider parameter space, RBL2 clearly
  improves 5- and 10-min extremes. GMM with standard estimators rather than block estimators matters,
  because block estimators under-estimate skewness by about 4% and that hurts extremes. Data: 69 years
  of 5-min rainfall at Bochum (Germany) and 105 years of 10-min data at Uccle (Belgium). —
  [Onof & Wang 2020](https://hess.copernicus.org/articles/24/2791/2020/)
- Cross, Onof, Winter & Bernardara (2018, HESS 22:727-...): "censored" BL calibration that models
  only the heavy part of the record, to estimate extremes at 5, 15 and 60 min. Cross et al. (2020,
  *Adv. Water Resour.* 136:103479) extend it to temperature-dependent censored simulation for future
  extremes. — [Cross et al. 2018](https://hess.copernicus.org/articles/22/727/2018/); [Northrop 2024 ref list](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)
- Park, Onof & Kim (2019, HESS 23:989-1014): a hybrid model reproducing rainfall characteristics from
  hourly to yearly scales, addressing RBL's under-estimation of sub-hourly and hourly extremes. —
  [HESS](https://hess.copernicus.org/articles/23/989/2019/)
- Kim & Onof (2020, *J. Hydrol.*): an RBLRP-based model tested on the 69-year Bochum 5-min record.
  Mean, variance, covariance, skewness and intermittency are reproduced "without any systematic bias"
  from 5 min to a decade, and extremes are reproduced from 5 min to 3 days. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169420306107) (abstract via search snippet; full text paywalled)
- Park, Cross, Onof, Chen & Kim (2021, *J. Hydrol.* 598:126296): "A simple scheme to adjust Poisson
  cluster rectangular pulse rainfall models for improved performance at sub-hourly timescales." —
  [Northrop 2024 ref list](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)
- A 2016 *J. Hydrol.* scheme couples a BL model with adjusting procedures to disaggregate to any
  sub-hourly scale. — [ScienceDirect S0022169416304450](https://www.sciencedirect.com/science/article/abs/pii/S0022169416304450)
- 2026: "Stochastic Poisson cluster simulation of sub-hourly precipitation under climate change in
  Germany and South Korea" (*Stoch. Environ. Res. Risk Assess.*). — [Springer](https://link.springer.com/article/10.1007/s00477-026-03182-6) (paywalled; only the title was verified)

**Software**
- pyBL v1.0.0, a Python package for BL modelling, with an application to short records (GMD 18:1357,
  2025). — [GMD](https://gmd.copernicus.org/articles/18/1357/2025/)
- NEOPRENE v1.0.1, a Python library for spatial rainfall based on the Neyman-Scott process (GMD 16:5035,
  2023). — [GMD](https://gmd.copernicus.org/articles/16/5035/2023/)
- RainSim (spatial-temporal NSRP), MOMFIT (single-site fitting), LetItRain (Kim et al.), and STORAGE
  (De Luca & Petroselli 2021, a user-friendly generator of long, high-resolution series). —
  [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)

### Inferences
- These models build event coherence in by construction (storms → overlapping cells), which is
  exactly what our first Gaussian AR baselines lacked. A fitted RBL2 at Astlingen would almost certainly not
  produce "75% single-step showers". It is the most natural strong classical baseline for us, and pyBL
  makes it cheap to run.
- The benchmark dataset of this literature (Bochum, 69 years of 5-min data) is in Germany, near our
  domain. A head-to-head comparison against RBL2 on our holdout period is defensible to a German or
  urban-hydrology audience.
- Fitting cost is a GMM optimisation per month (seconds to minutes), and simulation is near-instant.
  The computational advantage over diffusion is large.

### Gaps
- I did not find a quantitative benchmark of RBL2 against measured IDF at 5 min for a 10-year
  record. Results reported use 69-year or 105-year records.
- Foundational papers I did not open in this session: Rodriguez-Iturbe, Cox & Isham (1987, 1988),
  Onof et al. (2000) review "Rainfall modelling using Poisson-cluster processes" (*SERRA*), and the
  Wheater et al. reviews. Their citation details are not verified here.
- The quantitative results of the 2026 SERRA Germany/South Korea paper are paywalled.

---

## 2. Multiplicative random cascades / multifractal disaggregation

### Takeaway
Cascades disaggregate daily or hourly totals down to 5-10 min by repeatedly splitting mass with
random weights. Micro-canonical cascades conserve mass exactly, so long daily records can be turned
into 5-min series. They are cheap and good at scaling and the dry fraction. Their known weaknesses are
under-estimated autocorrelation, too many tiny sub-resolution intensities, and poor event structure
across block boundaries. They are widely used in German urban-hydrology research (Hannover group).

### Cited Findings
- Cascade approach: the average intensity of a coarse interval is redistributed to two finer
  intervals by random weights, repeatedly. Canonical variants (log-Poisson, log-Lévy, universal
  multifractal with as few as two parameters) are calibrated on the log-log scaling of sample
  q-moments. Olsson (1998, HESS 2:19-30) introduced dependence between cascade weights and an
  explicit treatment of zeros. — [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)
- Müller & Haberlandt (2018) built a micro-canonical cascade that disaggregates daily to 5 min. The
  micro-canonical version conserves rainfall exactly at each step. "A well-known problem of time
  series generated in this way is the inadequate representation of the autocorrelation." —
  [Müller-Thomy 2020, HESS 24:169](https://hess.copernicus.org/articles/24/169/2020/)
- Müller-Thomy (2020) tested two fixes on 24 stations. Lag-1 autocorrelation bias improved from about
  -4% (baseline) to about -3% (position-dependent first step) and to about 1% (dependence carried
  through all steps). Adding simulated-annealing resampling gave about 0.9%. Extremes for return
  periods of 1-5 years stayed acceptable, and scaling matched observations down to 120 min. Stated
  weaknesses: method C generates too many sub-resolution intensities, resampling is computationally
  demanding, and the study covers a single region. — [Müller-Thomy 2020](https://hess.copernicus.org/articles/24/169/2020/)
- Molnar & Burlando (2005, *Atmos. Res.* 77:137-151), "Preservation of rainfall properties in
  stochastic disaggregation by a simple random cascade model". As cited in later work, disaggregation
  from quasi-daily values produced a 48% fraction of 10-min intensities below the measurement
  resolution. — [search result citing the paper](https://hess.copernicus.org/articles/24/169/2020/) (figure taken from a secondary citation, not checked in the primary)
- Lombardo, Volpi & Koutsoyiannis (2012, *Hydrol. Sci. J.*) compared multifractal and
  Hurst-Kolmogorov discrete random cascades for downscaling in time. — [T&F](https://www.tandfonline.com/doi/full/10.1080/02626667.2012.695872) (title verified; content not read)
- An evaluation of six cascade models plus a new generator for urban hydrology appeared in
  *Atmospheric Research* (2011). — [ScienceDirect S0169809510003510](https://www.sciencedirect.com/science/article/abs/pii/S0169809510003510) (title verified)
- Temperature-dependent cascade disaggregation of climate-model output is used to estimate future
  sub-daily extremes (NHESS 24:2025, 2024). — [NHESS](https://nhess.copernicus.org/articles/24/2025/2024/) (title verified)
- Regionalising cascade parameters for ungauged sites: SERRA 2026. — [Springer](https://link.springer.com/article/10.1007/s00477-026-03360-6) (title verified)

### Inferences
- Cascades need a coarse driving series (observed daily data or a daily generator). For a 10-year
  synthetic series they are usually paired with a daily occurrence-amount generator, which adds a
  second model to validate.
- The under-estimated ACF and the excess of tiny intensities are close cousins of our AR baselines'
  fragmentation problem, though less severe, because the mass of each hour or day stays in place.

### Gaps
- I found no 2015-2026 head-to-head paper that scores cascades, BL and a learned generator on the
  same 5-min IDF.

---

## 3. Occurrence-amount, copula, CoSMoS, HMM, resampling, and alternating-renewal models

### Takeaway
A broad family separates occurrence (Markov chain, HMM, wet/dry renewal) from amount (parametric
distribution, copula, or resampling). At sub-hourly scales the most relevant to German urban drainage
are: the alternating-renewal model (Hannover group, 5 min, 800 German stations); copula/Markov
disaggregators (WayDown); CoSMoS, which transforms a parent Gaussian process to match marginals,
ACF and intermittency; and non-parametric method-of-fragments or k-NN resampling. HMMs are used mainly
at hourly to daily resolution.

### Cited Findings
**Alternating renewal (event-based) models**
- Alternating-renewal models (sequences of independent wet and dry periods, with event intensity
  and profile models) by Callau Poduje & Haberlandt have been "successfully applied at the 5 min
  timestep for urban applications". The model was later extended to multiple sites. In a 2022
  erosivity comparison (800 German stations, 5-min data, 10-25-year records), regionalising the 5-min
  data beat rainfall generation models including the alternating renewal model (ARM). —
  [ESurf 10:851, 2022](https://esurf.copernicus.org/articles/10/851/2022/)
- Menabde & Sivapalan (2000) combined a coarse alternating-renewal model with a fine-scale
  multiplicative cascade. — [HESS 2024 search snippet](https://hess.copernicus.org/articles/28/391/2024/)
- A vine-copula approach for spatio-temporal synthesis of continuous precipitation series (*Water*
  10:862, 2018). — [MDPI](https://doi.org/10.3390/w10070862) (title verified)

**Copula / Markov disaggregation, and double-Poisson generator: comparison including Germany**
- Vorobevskii, Park, Kim, Barfus & Kronenberg (2024, HESS 28:391) compared WayDown (first-order
  Markov chains plus copulas; conditional; preserves daily sums) with LetItRain (double Poisson
  process; unconditional) at 10 min. They used five German stations (Leipzig, Naumburg, Greiz, Hof,
  Klingenthal) and five Korean ones, and applied climate change through change factors from
  projections. Both models over-estimate up to a threshold and under-estimate above it. **Annual
  maxima are systematically under-predicted at German sites.** WayDown's ACF is good at lag 1 but too
  low at longer lags, and LetItRain captures higher lags better. Dry proportion was within 1-3% for
  LetItRain in Germany. LetItRain struggled with seasonal cycles and high-variability months. —
  [HESS 2024](https://hess.copernicus.org/articles/28/391/2024/)

**CoSMoS (Papalexiou)**
- Papalexiou (2018, *Adv. Water Resour.* 115:234-252), "Unified theory for stochastic modelling of
  hydroclimatic processes: preserving marginal distributions, correlation structures, and
  intermittency". Any process is obtained by transforming a parent Gaussian process. It was extended to
  random fields by Papalexiou & Serinaldi (2020). The CoSMoS R package preserves "any probability
  distribution and any linear autocorrelation structure", and a Python port (PyCoSMoS) exists. —
  [CRAN docs](https://rdrr.io/cran/CoSMoS/man/CoSMoS-package.html); [USask](https://water.usask.ca/resources/cosmos.php); [PyCoSMoS](https://www.researchgate.net/publication/380708576_PyCoSMoS_An_advanced_toolbox_for_simulating_real-world_hydroclimatic_data)

**HMMs**
- Stoner & Economou (2020, *Comput. Stat. Data Anal.*, doi:10.1016/j.csda.2020.107045): a Bayesian
  HMM for **hourly** rainfall with clone states and non-stationary transitions. It targets long dry
  periods, seasonal occurrence and intensity, and extremes, and is demonstrated on 8 years of hourly
  data for urban flood modelling. — [arXiv 1906.03846](https://arxiv.org/abs/1906.03846)
- An interpretable seasonal multisite HMM for stochastic rain generation in France (ASCMO 11:159,
  2025; daily). — [ASCMO](https://ascmo.copernicus.org/articles/11/159/2025/) (title verified)

**Non-parametric resampling**
- Method of fragments (MOF): resample a vector of sub-daily fractions of a daily total, usually
  conditioned on k-NN similarity. Pui et al. (2012, *J. Hydrol.* 470:138-157) compared daily-to-sub-daily
  disaggregation alternatives. Westra et al. (2012, WRR 48:W01535) and Mehrotra et al. (2012, WRR
  48:W01536) built a regionalised sub-daily disaggregation and daily generation chain. Li et al. (2018,
  *Int. J. Climatol.*) proposed three MOF resampling variants. — [Westra et al. 2012](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2011WR010489); [Li et al. 2018](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.5438)
- Pyraingen (2024, *Env. Modelling & Software*) is a Python package for constrained continuous
  rainfall generation implementing these Australian approaches. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1364815224000458)
- k-NN with gamma-kernel perturbation produces new values rather than reshuffling. — [ResearchGate](https://www.researchgate.net/publication/273022833_Precipitation_Simulation_Based_on_k-Nearest_Neighbor_Approach_Using_Gamma_Kernel)
- A semi-parametric hourly space-time weather generator couples an hourly rainfall model with k-NN
  resampling of other variables (HESS 27:3957, 2023). — [HESS](https://hess.copernicus.org/articles/27/3957/2023/)

### Inferences
- CoSMoS is the closest relative of our latent Gaussian AR baseline: a transformed parent Gaussian
  with a target ACF and intermittency. Our first latent Gaussian fit fragmented rain because its parent
  correlation was under-estimated (lag-1 0.47; occurrence needs 0.976). Refitted with the parent
  correlation matched to occurrence at every lag up to 5h20 (2026-10-05), it reproduces long storms
  (15.8/yr vs 14.3) but over-produces 1-24 h extremes: one parent process carries both occurrence
  and amount, so heavy rain persists. This suggests CoSMoS-style generators at 5 min get event
  timing from a well-fitted parent ACF but need separate amount dependence for extremes. This is our
  inference, not a published finding.
- Alternating-renewal models describe whole events (duration, depth, internal profile), which is the
  property RL training episodes most need. They are a natural "event-structured" classical baseline.

### Gaps
- I found no published CoSMoS application at 5-min resolution for urban drainage.
- Callau Poduje & Haberlandt's original single-site ARM paper (I believe *J. Hydrol.* 2017) was not
  opened. Its title and venue are unverified here.

---

## 4. German practice: KOSTRA, design storms, NiedSim

### Takeaway
German urban-drainage practice uses (a) KOSTRA-DWD IDF grids for design storms, for example Euler
type II hyetographs, and (b) for continuous simulation, measured long series or NiedSim3 synthetic
5-min series. NiedSim3 is an operational resampling and simulated-annealing generator distributed by
four state authorities, which post-hoc rescales extremes to KOSTRA. It is the most directly relevant
"practice" comparator for Astlingen.

### Cited Findings
- KOSTRA-DWD (Koordinierte Starkniederschlagsregionalisierung und -auswertung) is DWD's heavy-rain
  catalogue. KOSTRA-DWD-2020 has been the valid version since 1 January 2023, replacing KOSTRA-DWD-2010R,
  on a 5 km × 5 km ETRS89-LAEA grid with durations from D = 5 min. Tools generate Euler-type design
  storms from it. — [DWD user guide](https://dwd.de/DE/leistungen/kostra_dwd_rasterwerte/download/kostra_dwd_2020_anwenderhilfe_pdf.pdf?__blob=publicationFile&v=3); [itwh](https://itwh.de/software/niederschlag/auswertung-von-niederschlagsdaten/itwh-kostra-dwd-2020/)
- Design of drainage networks targets a flooding frequency for a 1-10-year design rain (DWA-A118).
  Combined-sewer overflow assessment (DWA-A102 draft, ATV-A128) needs **continuous long-term
  simulation**, because overflows occur several times a month and depend on wet/dry alternation and
  antecedent filling. — [Müller, Mosthaf & Bárdossy 2018, HyWa 62(4)](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf)
- NiedSim (Bárdossy 1998) has been distributed since 2000 in Baden-Württemberg, since 2003 in Hesse
  and Rhineland-Palatinate, and since 2009 in Bavaria. Engineering firms order synthetic series for any
  1 km grid cell. NiedSim3 was released in August 2017 after the BMBF projects SYNOPSE and SAMUWA. —
  [Müller et al. 2018](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf)
- **NiedSim3 algorithm**, all from [Müller et al. 2018](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf):
  - **Database:** statistics from 1,034 5-min, 1,551 hourly and 5,468 daily stations (1961-2012) are
    kriged to a 1 km grid.
  - **Hourly generation, year by year:** values are drawn from seasonal hourly distributions until the
    annual total is reached. About 80% of large values (> 4 mm/h) are fixed in place by a seasonal
    Poisson process, so they cannot merge into unrealistic mega-events.
  - **Hourly optimisation:** simulated annealing swaps hourly values to match lag-1 ACF at 1-24 h
    aggregations, monthly totals, daily exceedance probabilities (0, 1, 5 mm), and the
    wet-spell-length distribution.
  - **Disaggregation to 5 min:** each hour is first spread uniformly. Simulated annealing then shifts
    0.01-0.1 mm between 5-min steps within the hour, conserving mass, to match lag-1 and lag-2 ACF at
    5-60 min and the 1st-3rd moment scaling across 5-60 min, per season.
  - **Extreme adjustment:** values in the partial series for durations 5, 15, 30, 60, 120, 360, 720
    and 1,440 min are rescaled to a chosen IDF (operationally KOSTRA-DWD-2010R), keeping the temporal
    structure and mass. This decouples extremes from the continuous structure.
- NiedSim3 validation: synthetic series match observed statistics, and **combined-sewer simulation
  results fall within the bootstrap sampling uncertainty of observed series**, when interpolated
  parameters match local statistics. Under cross-validation in orographically heterogeneous areas
  there are systematic deviations. Homogeneous 5-min weighing-gauge records are under 20 years long
  because DWD introduced weighing gauges from the mid-1990s. Gauge type (digitised chart, tipping
  bucket, weighing) visibly changes 5-min ACF, scaling and wet-interval statistics. — [Müller et al. 2018](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf)
- Related Stuttgart work: "Generation of a realistic temporal structure of synthetic precipitation
  time series for sewer applications" (Müller et al.). — [ResearchGate](https://www.researchgate.net/publication/324561578_Generation_of_a_Realistic_Temporal_Structure_of_Synthetic_Precipitation_Time_Series_for_Sewer_Applications) (title verified)

### Inferences
- NiedSim3's two-stage design (structure by optimisation, then IDF-rescaling of extremes) is a
  strong idea we could borrow as a post-processing step on diffusion output, or use as a baseline.
  Its sewer-model validation within the bootstrap uncertainty of observations is the same kind of
  downstream test as our planned agent-in-the-loop milestone.
- Astlingen is a benchmark catchment in BW/German literature, so a reviewer may ask "why not
  NiedSim/KOSTRA?". A comparison table should place our model against KOSTRA-DWD-2020 IDF values for
  the Astlingen grid cell.

### Gaps
- I did not verify whether NiedSim is available for research outside state engineering use, or its
  licence terms.

---

## 5. Lengths, resolutions, long storms, extremes, seasonality, climate change, cost

### Takeaway
Classical generators handle arbitrary lengths (100s-1000s of years) at negligible cost. Seasonality is
usually handled by fitting per month or season. Climate change is handled by change factors,
temperature scaling or censored refits. Extremes at 5 min are the persistent weak spot, addressed
by adding skewness to fits, heavier cell distributions, censoring, or post-hoc IDF rescaling.

### Cited Findings
- Seasonality: monthly GMM fits are standard for Poisson-cluster models — [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf). NiedSim uses summer/winter distributions and season-weighted ACF — [Müller et al. 2018](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf).
- Length: RBL fits reproduce statistics from 5 min to a decade on 69 years of Bochum data — [Kim & Onof 2020](https://www.sciencedirect.com/science/article/abs/pii/S0022169420306107). NiedSim generates "time series of any length", homogeneous and gap-free — [Müller et al. 2018](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf).
- Climate change: change factors applied to statistics — [Vorobevskii et al. 2024](https://hess.copernicus.org/articles/28/391/2024/). Temperature-dependent censored BL — [Cross et al. 2020, via Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf). Temperature-dependent cascade disaggregation — [NHESS 2024](https://nhess.copernicus.org/articles/24/2025/2024/). NiedSim's IDF step can be swapped for future IDF without re-optimising structure — [Müller et al. 2018](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf).
- Long storms and multi-day structure: Poisson-cluster models cannot produce negative ACF and tend to
  over-estimate daily extremes while under-estimating sub-hourly ones — [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf); [Onof & Wang 2020](https://hess.copernicus.org/articles/24/2791/2020/).
- Cost: simulated annealing resampling is "computationally demanding" — [Müller-Thomy 2020](https://hess.copernicus.org/articles/24/169/2020/). WayDown is expensive at high precision — [Vorobevskii et al. 2024](https://hess.copernicus.org/articles/28/391/2024/).

### Inferences
- Against these baselines, the diffusion model's possible advantages are joint realism of event shape
  across many statistics without hand-picked objective terms, and conditioning on regime (HMM state).
  Its disadvantages are cost, data-hunger (10-year record vs. 69-year Bochum fits), and no analytic
  guarantees.

### Gaps
- I found no published wall-clock comparisons between classical generators and deep generative
  rainfall models.

---

## 6. Evaluation criteria reported in this literature

### Takeaway
There is no single standard. Papers report a mix of moments across aggregation scales, dry
proportion, ACF, extreme-value or IDF statistics, event statistics, and, increasingly, downstream
hydraulic validation (sewer-model outputs within the bootstrap uncertainty of observations).

### Cited Findings
- No unified validation procedure exists. The literature uses ACF (Pui et al. 2012), distribution
  functions (Paschalis et al. 2014), extreme-value statistics (Koutsoyiannis et al. 2011), the
  probability of zero rain p0 (Molnar et al. 2005), and event statistics such as duration, frequency
  and volume (Vandenberghe et al. 2011). Indirect validation through sewer simulation is recommended
  and is rare (Gaume et al. 2007; Andrés-Doménech et al. 2010; Müller et al. 2016). Acceptable
  deviations are defined relative to the sampling variability of observed series via bootstrap. —
  [Müller et al. 2018](https://doi.bafg.de/HyWa/2018/HyWa_2018,4_1.pdf)
- Standard GMM fitting and validation sets: Set 1 (MOMFIT manual) is the 1-h mean plus variance,
  lag-1 ACF, skewness and proportion wet. Set 2 uses these at 10 min, 1 h, 6 h and 24 h. Simulated
  extremes are checked by simulation because there are no closed forms. — [Northrop 2024](https://discovery.ucl.ac.uk/10174912/1/StochasticModelsOfRainfall.pdf)
- Onof & Wang (2020) compare return-level estimates across durations (5 min to daily) against
  observed annual maxima — [HESS](https://hess.copernicus.org/articles/24/2791/2020/). Müller-Thomy (2020) uses lag-1 ACF bias, 1-5-year return-period extremes, and moment scaling — [HESS](https://hess.copernicus.org/articles/24/169/2020/). Vorobevskii et al. (2024) use quantile plots of extremes, annual maxima, ACF at several lags, and dry proportion — [HESS](https://hess.copernicus.org/articles/28/391/2024/).

### Inferences
- Our Gate A criteria (IDF cells in band, wet/dry spells, event statistics, ACF) are in line with this
  literature. Bootstrap bands from the observed record, as NiedSim3 used, are the accepted way to set
  "in band" thresholds, and our agent-in-the-loop milestone matches the "indirect validation via sewer
  simulation" tradition.

### Gaps
- I found no review paper (2015-2026) that benchmarks all classical families on a common 5-min
  dataset with a common metric set. Northrop (2024) is the most recent broad review found, but it
  focuses on mechanistic point-process models.
