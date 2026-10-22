# Long time-series generation with diffusion models, horizon effects, and block-bootstrap block length

Notes compiled 2026-10-22. Ratings use two axes for every relevant work:
- **(a)** closeness to the thesis's *window-ceiling / independent-window oracle* point ("a generator that only models dependence inside a W-step window cannot exceed what independently glued real W-step windows achieve at longer scales, and a seasonal block bootstrap of real windows predicts that ceiling").
- **(b)** closeness to the *decision-aware evaluation* headline ("a scenario generator only needs to be accurate where the decision-makers disagree").

Scale: SAME IDEA / PARTIAL / DIFFERENT. Citations marked **[not re-opened]** are given from prior knowledge with a standard link; their bibliographic details were not checked again in this session and should be confirmed before they go into the paper.

---

## Q1. How do ImagenTime, ImagenFew, Diffusion-TS, TSDiff, CSDI, TimeGrad and 2024-2026 methods generate long series, and do any report long-range statistics beyond the window?

### Takeaway
The time-series diffusion literature handles length in four ways: a bigger image or tensor covering the whole series (ImagenTime), fixed short windows with no extension (ImagenFew, Diffusion-TS, CSDI), autoregression with a recurrent state (TimeGrad), and stitching or assembling windows (WaveStitch, Long et al. 2026). No paper found evaluates statistics at lags or aggregation scales longer than the generator's window. The closest, Long et al. (Oct 2026), checks autocorrelation only at a lag inside its window.

### Cited Findings
**ImagenTime (Naiman et al., NeurIPS 2024)**: "Utilizing Image Transforms and Diffusion Models for Generative Modeling of Short and Long Time Series" — [arXiv 2410.19538](https://arxiv.org/abs/2410.19538); [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/dc6748383752138af7f00b3185a0a404-Paper-Conference.pdf)
- The method uses invertible transforms ("delay embedding and the short-time Fourier transform") so that it can "process short- and long-range inputs within the same framework". — [arXiv 2410.19538](https://arxiv.org/abs/2410.19538)
- Long-term benchmarks (FRED-MD, NN5 Daily, Temperature Rain) are about 750 steps, each turned into one 32×32 STFT image. The ultra-long benchmarks are Traffic (17,544 steps; delay embedding with n=144, m=136, giving **144×144 images**) and KDD-Cup (10,920 steps; 112×112 STFT images). — [arXiv HTML](https://arxiv.org/html/2410.19538)
- Each whole sequence appears to be mapped to one image, with no patching or stitching described. The paper says "The delay embedding scales naturally to long sequences", excludes the Gramian angular field because it "produces huge images", and analyses no long-range statistics. The long datasets were picked by lag-1 autocorrelation (0.38 to 0.98). — [arXiv HTML](https://arxiv.org/html/2410.19538)

**ImagenFew (Gonen et al., NeurIPS 2025)**: "Time Series Generation Under Data Scarcity: A Unified Generative Modeling Approach" — [arXiv 2505.20446](https://arxiv.org/abs/2505.20446v1); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115993)
- Pre-trains four separate models at fixed lengths of 12, 24, 36 and 64 steps; the main results use 24. Delay embedding is zero-padded to a square image. A dynamic binary mask lets a model trained at one length fine-tune to others. — [arXiv HTML](https://arxiv.org/html/2505.20446v1)
- No stitching, no generation beyond the trained length, and no long-range statistics. At length 64, "the model pre-trained on that specific length substantially outperforms those pre-trained on shorter sequences." — [arXiv HTML](https://arxiv.org/html/2505.20446v1)

**Other time-series diffusion models [not re-opened]**
- Diffusion-TS (Yuan & Qiao, ICLR 2024) generates fixed-length windows; its "long-term" experiments use windows of 64/128/256 steps, each generated in one pass. — [arXiv 2403.01742](https://arxiv.org/abs/2403.01742)
- TSDiff (Kollovieh et al., NeurIPS 2023) is an unconditional diffusion model trained on fixed context-plus-horizon windows, with "observation self-guidance" for forecasting or imputation at inference. Conditioning is inference-time guidance, the same family as the thesis's inpainting continuation. — [arXiv 2307.11494](https://arxiv.org/abs/2307.11494)
- CSDI (Tashiro et al., NeurIPS 2021) is a conditional score model for imputation on fixed windows, trained with self-supervised masking. It is the time-series precedent for the thesis's "trained with known prefixes" continuation. — [arXiv 2107.03502](https://arxiv.org/abs/2107.03502)
- TimeGrad (Rasul et al., ICML 2021) is an autoregressive RNN whose per-step diffusion head is conditioned on the hidden state. It can produce arbitrary length, but long-range memory has to pass through the RNN state. — [arXiv 2101.12072](https://arxiv.org/abs/2101.12072)
- MG-TSD (Fan et al., ICLR 2024) uses multi-granularity (coarse-to-fine) guidance for forecasting diffusion. It is a precedent for the multi-resolution / two-resolution idea. — [arXiv 2403.05751](https://arxiv.org/abs/2403.05751) [not re-opened]

**2024-2026 approaches to longer-than-window generation**
- **WaveStitch** (SIGMOD 2026; TU Delft / Neuchâtel) generates windows *in parallel* and keeps them coherent with an inference-time conditional loss and a stitching mechanism. It frames the trade-off as "autoregressive methods are slow but coherent, parallel methods fast but can be incoherent across windows". It reports up to 166.48× faster than autoregressive methods in v2; earlier versions give different numbers. — [arXiv 2503.06231](https://arxiv.org/pdf/2503.06231); [v2 HTML](https://arxiv.org/html/2503.06231v2)
- **Long et al. (arXiv 2610.10222, 7 Oct 2026; verified)**: "Evaluating Sequence Assembly Strategies for Differentially Private Synthetic Time-Series Forecasting" by Guoxiong Long, Huizhen Huang, Qikun Cai, Tao Huang and Chen Hou. The generator is "a DP-SGD-trained conditional 1D U-Net diffusion model that produces fixed-length synthetic windows" (7-day windows: L=168/672/1008), assembled into 70-day sequences with varied overlap rates and window weighting. — [arXiv abs](https://arxiv.org/abs/2610.10222); [HTML](https://arxiv.org/html/2610.10222)
  - Findings: "Increased overlap generally improves boundary continuity", but "improvements in continuity or individual fidelity diagnostics do not consistently reduce forecasting error". ARIMA and Prophet are more sensitive to "the temporal changes introduced by aggressive blending". — [HTML](https://arxiv.org/html/2610.10222)
  - The only temporal metric is a **24-hour autocorrelation deviation, a lag inside the 7-day window**. There is **no real-window assembly oracle**, only Train-on-Real-Test-on-Real (TRTR) forecasting references. Block bootstrap is not mentioned in the text that was read (about 80% of the paper; the reference list was truncated). — [HTML](https://arxiv.org/html/2610.10222)
- **T2S** (IJCAI 2025) uses a length-adaptive VAE with a diffusion-transformer denoiser and generates "arbitrary lengths" from a shared latent space. — [arXiv 2505.02417](https://arxiv.org/pdf/2505.02417)
- **Block diffusion (BD3-LM, ICLR 2025)**, a language-model method, runs diffusion inside a block and conditions each block autoregressively on earlier blocks, which allows sequences longer than the training context. — [arXiv 2503.09573](https://arxiv.org/pdf/2503.09573)
- **Spatiotemporal turbulence (Bayesian conditional diffusion)**: because memory limits the training window, it uses autoregressive conditional sampling to produce flows of arbitrary temporal length. — [arXiv 2311.07896](https://arxiv.org/pdf/2311.07896)

### Inferences
- The thesis's design space maps onto existing families: one-day 24×24 window = ImagenTime-style bigger image; crossfade/bridging = Long et al. / WaveStitch-style assembly; inpainting continuation = TSDiff-style guidance / replacement; known-prefix training = CSDI-style masked training; two-resolution = MG-TSD-style multi-granularity. The method list is therefore not novel. What is new is the **diagnostic**: measuring, and bounding with a real-window oracle, the statistics at scales beyond the window.
- ImagenTime's own scaling rule (image side ≈ √L) makes a single-image 10-year 5-minute series (~1.05 M steps) need about a 1,026×1,026 image. That is computational support for why windowing is unavoidable here, and a sentence worth putting in the paper (arithmetic, not from a source).
- Long et al. is the closest recent paper and the one the thesis must cite and set itself apart from. It finds, as the thesis does, that boundary continuity does not predict downstream utility. It does **not** test a real-window oracle, statistics beyond the window, or a decision-specific error metric.

**Closeness ratings**
| Work | (a) window-ceiling / oracle | (b) decision-aware headline |
|---|---|---|
| ImagenTime 2024 | DIFFERENT: avoids the ceiling by making the image cover the whole series and never measures beyond-window statistics. | DIFFERENT: evaluated on generic fidelity and classification scores. |
| ImagenFew 2025 | DIFFERENT: fixed lengths ≤64, no extension. | DIFFERENT: generic benchmarks. |
| Diffusion-TS 2024 | DIFFERENT: one-pass fixed windows, up to 256. | DIFFERENT. |
| TSDiff 2023 | DIFFERENT: guidance-based conditioning within a fixed window, no beyond-window analysis. | DIFFERENT: forecasting accuracy, not decision disagreement. |
| CSDI 2021 | DIFFERENT: imputation within windows. | DIFFERENT. |
| TimeGrad 2021 | DIFFERENT: autoregressive, no horizon analysis. | DIFFERENT. |
| WaveStitch 2025/26 | DIFFERENT: fixes boundary coherence, not long-range statistics. | DIFFERENT. |
| Long et al. 2026 | PARTIAL: assembles fixed diffusion windows into long series and varies overlap, but has no real-window oracle and no beyond-window statistic. | PARTIAL: judges assembly by downstream TSTR forecasting utility and finds that continuity does not predict utility, but it is task utility, not disagreement between decision-makers. |
| T2S / BD3-LM / turbulence AR | DIFFERENT: length-extension methods with no ceiling analysis. | DIFFERENT. |
| MG-TSD | DIFFERENT: multi-resolution for forecasting, no ceiling analysis. | DIFFERENT. |

### Gaps
- I did not read Diffusion-TS, TSDiff, CSDI or TimeGrad again this session. The window lengths quoted are from prior knowledge.
- I could not confirm whether WaveStitch evaluates any statistic beyond the window. The search snippet only described coherence and MSE.
- The last ~20% of Long et al. (including the reference list) was not read, so a block-bootstrap citation there cannot be fully ruled out.
- I found no 2024-2026 time-series diffusion paper that reports autocorrelation, run lengths or variance of aggregates at scales longer than the training window.

---

## Q2. Autoregressive and continuation diffusion in other domains (video, inpainting): drift, RePaint biases, mode-seeking, and zero-inflated data

### Takeaway
Video diffusion has a well-documented result that matches the thesis's inpainting finding in kind. **Replacement-based conditioning (the RePaint / Song et al. imputation trick) ignores the term that couples the generated block to the known block, so block-autoregressive extension produces blocks that "appear to be uncorrelated samples".** That is, it falls back to independent-window behaviour (Ho et al. 2022). Fixes are trained conditioning (Diffusion Forcing, DFoT) and guidance on history (History Guidance), but CFG-style guidance has its own costs. **No work found analyses replacement or guidance bias on zero-inflated or intermittent data.** The thesis's "inpainting halves rainfall" result appears to be new.

### Cited Findings
- **Ho et al., "Video Diffusion Models" (NeurIPS 2022)** define the *replacement method*: known-region latents are overwritten with forward-process samples at each step. They found "it to not work well for our video models: Although samples x_b looked good in isolation, they were often not coherent with x_a". The cause: latents are updated toward E[x_b | z_t] when what is needed is E[x_b | z_t, x_a] = E[x_b | z_t] + (σ_t²/α_t) ∇ log q(x_a | z_t), "where the second term is missing in the replacement method". Their fix is **reconstruction guidance**. — [arXiv 2204.03458](https://arxiv.org/abs/2204.03458) (quotes taken from the PDF text)
- In block-autoregressive long-video generation, Ho et al. report that "The samples from the replacement method clearly show a lack of temporal coherence, since frames from different blocks throughout the generated videos appear to be uncorrelated samples (conditioned on c)". — [arXiv 2204.03458](https://arxiv.org/abs/2204.03458)
- **RePaint (Lugmayr et al., CVPR 2022)** is replacement conditioning plus resampling ("jump back") steps. — [arXiv 2201.09865](https://arxiv.org/abs/2201.09865) [not re-opened]
- A theoretical analysis of RePaint finds that its replacement sampling has a **bias caused by misalignment** that "hampers sample recovery even in a two-state diffusion process". Even "least-squares projections onto the data manifold do not recover the original samples". The proposed RePaint+ corrects the drift and dispersion terms. — [arXiv 2302.01217](https://arxiv.org/abs/2302.01217) ("A Theoretical Justification for Image Inpainting using Denoising Diffusion Probabilistic Models"; authors believed to be Rout, Parulekar, Caramanis & Shakkottai [not re-opened])
- A guided-inpainting paper explains that RePaint puts the noised known region next to the generated region, and "the inherent noise mismatch… leads to semantic inconsistency". — [GuidPaint, arXiv 2507.21627](https://arxiv.org/pdf/2507.21627)
- Exact conditional sampling from an unconditional diffusion model is possible asymptotically with sequential Monte Carlo (Twisted Diffusion Sampler), which shows that the common heuristics are approximations. — [Wu et al., NeurIPS 2023, arXiv 2306.17775](https://arxiv.org/abs/2306.17775) [not re-opened]
- **Diffusion Forcing (Chen et al., NeurIPS 2024)** trains with an independent noise level per token, so a causal model can denoise future tokens conditioned on a clean or partly noisy past. It supports "rolling-out sequences of continuous tokens, such as video, with lengths past the training horizon, where baselines diverge." — [arXiv 2407.01392](https://arxiv.org/abs/2407.01392) (the arXiv page names no venue; NeurIPS 2024 is from prior knowledge)
- **History-Guided Video Diffusion (Song, Chen, Simchowitz, Du, Tedrake, Sitzmann; ICML 2025)** introduces the Diffusion Forcing Transformer and history guidance (CFG with history as the condition). Key quotes: "CFG-style history dropout performs poorly"; "vanilla history guidance, already significantly improves video generation quality and temporal consistency"; "history guidance across time and frequency further enhances motion dynamics"; the method "can stably roll out extremely long videos". — [arXiv 2502.06764](https://arxiv.org/abs/2502.06764)
- **Classifier-free guidance** trades diversity for fidelity as the guidance weight grows. — [Ho & Salimans 2022, arXiv 2207.12598](https://arxiv.org/abs/2207.12598) [not re-opened]
- Other long-video schemes: FIFO-Diffusion (diagonal denoising queue for infinite video; [arXiv 2405.11473](https://arxiv.org/abs/2405.11473)), Rolling Diffusion (noise rising along time inside a sliding window; [arXiv 2402.09470](https://arxiv.org/abs/2402.09470)), Self Forcing (trains on the model's own rollouts to close the train/test gap, i.e. exposure bias, in autoregressive video diffusion; [arXiv 2506.08009](https://arxiv.org/abs/2506.08009)), and MultiDiffusion (joint denoising of overlapping windows fused by averaging, the image-domain analogue of crossfade; [arXiv 2302.08113](https://arxiv.org/abs/2302.08113)). [all not re-opened]
- **DiffCollage (Zhang, Song, Huang, Chen, Liu; CVPR 2023)** models long or large content as a factor graph: "each factor node represents a portion of the content", joined through overlap variables, and generated in parallel "without resorting to an autoregressive generation procedure". — [arXiv 2303.17076](https://arxiv.org/abs/2303.17076)
- **Zero-inflated data**: the only diffusion-plus-zero-inflation work found is a *forecasting* pipeline (ZIDF). It adds Gaussian noise to smooth the zero-inflated target, then denoises, and the search showed no dry-fraction calibration analysis. — [arXiv 2509.10501](https://arxiv.org/abs/2509.10501)

### Inferences
- Ho et al.'s derivation fits the thesis directly. Replacement-style continuation drops ∇ log q(prefix | z_t), so the continuation is coupled to the prefix only weakly. Ho et al. saw this as blocks behaving like *uncorrelated samples*, the independent-window regime the thesis's oracle describes. Citing it explains why inpainting continuation cannot be expected to break the window ceiling.
- On mode-seeking: with zero-inflated rain, the posterior-mean denoiser at moderate noise favours "dry" when the prefix is ambiguous. Combined with guidance (which trades diversity for fidelity), this plausibly explains why inpainting halves rainfall. **This mechanism is the thesis's own hypothesis and is not found in the literature.** It should be presented as a finding plus a hypothesis, and the authors should not claim it is known.
- History Guidance is the closest prior work to the thesis's "classifier-free-style guidance" continuation. Its observation that vanilla guidance needed a frequency-domain variant to restore motion dynamics hints at the same damping effect, but this is an inference from the abstract, which does not say "static" or "damped".
- DiffCollage's factor-graph assembly implies that, beyond the overlap, pieces interact only through chains of overlaps (a Markov-in-overlap structure; my reading, not stated on the abstract page). That is a structural version of the window ceiling, though the paper does not frame or measure it that way.

**Closeness ratings**
| Work | (a) window-ceiling / oracle | (b) decision-aware |
|---|---|---|
| Ho et al. 2022 (replacement vs reconstruction guidance) | PARTIAL: shows replacement continuation falls back to near-independent blocks, the regime the oracle bounds, but gives no oracle or long-range measurement. | DIFFERENT. |
| RePaint 2022 / RePaint+ 2023 | PARTIAL (mechanism only): documents bias in replacement conditioning; no window or long-scale analysis. | DIFFERENT. |
| Twisted Diffusion Sampler 2023 | DIFFERENT: exact conditional sampling method. | DIFFERENT. |
| Diffusion Forcing 2024 | DIFFERENT: a method to roll out beyond the horizon; no ceiling analysis. | DIFFERENT (applies to planning, but does not evaluate generators by decisions). |
| History-Guided Video Diffusion 2025 | DIFFERENT for the ceiling; it is the closest prior work for the CFG-continuation variant. | DIFFERENT. |
| FIFO / Rolling / Self Forcing / MultiDiffusion | DIFFERENT: extension and drift methods. | DIFFERENT. |
| DiffCollage 2023 | PARTIAL: assembly via overlap factor graph implies limited dependence beyond the overlap, not analysed as a ceiling. | DIFFERENT. |
| ZIDF 2025 | DIFFERENT: forecasting, not continuation. | DIFFERENT. |

### Gaps
- **No paper found analyses mode-seeking or rainfall-volume loss from inpainting, replacement or guidance continuation on zero-inflated or intermittent data.** Searches on diffusion + zero-inflated/intermittent precipitation returned only forecasting work.
- History Guidance's maximum rollout length and its exact statement of the vanilla-CFG failure were not visible on the abstract page.
- I did not verify a quantitative "drift versus rollout length" result for long video (FIFO, Self Forcing). Those citations are given only as examples of methods.

---

## Q3. Does any work bound what an independent-window generator can achieve beyond its window, or use a real-data block bootstrap as an "oracle" upper bound for windowed generators?

### Takeaway
**The mathematical ceiling is classical; using it as an oracle for a learned generator appears to be new.** Block-bootstrap theory shows that resampling blocks of length ℓ underestimates the long-run variance of sums by about G/ℓ, where G = Σ_k |k| r(k). This holds for every standard block scheme. So the bootstrap reproduces only dependence shorter than the block, and a fixed ℓ caps it (Künsch 1989; Lahiri 1999; Nordman 2009). Hydrology and weather-generator work also documents that block and short-memory resamplers understate persistence and aggregate variability. I found **no ML paper that uses a real-window block bootstrap as an upper-bound oracle for a windowed deep generator, or that shows a trained model's long-scale failures match that oracle.** Long et al. (2026), the obvious candidate, does not.

### Cited Findings
- For the estimator of σ_n² = n·Var(X̄_n), the **bias is −G/ℓ + o(1/ℓ) with G = Σ_{k∈Z} |k| r(k)** (shown for the stationary bootstrap). Lahiri found that MBB, NBB, CBB and SB "all four bootstrap methods have the same asymptotic bias but their asymptotic variances can differ". The variance behaves like C·ℓ/n. — [Nordman 2009, Ann. Statist. 37(1):359-370, DOI 10.1214/07-AOS567](https://arxiv.org/abs/0903.0474) (quotes taken from the PDF text)
- Nordman also states that the stationary bootstrap's variance matches that of the non-overlapping block bootstrap, which corrects the earlier view that it was largest, and that the CBB's MSE-optimal block length is (3/2)^{1/3} times the SB's. — [Nordman 2009](https://arxiv.org/abs/0903.0474)
- The matched-block bootstrap (Carlstein, Do, Hall, Hesterberg & Künsch 1998) and the tapered block bootstrap (Paparoditis & Politis) "have similar order variances but smaller biases"; the matched-block version "requires stronger assumptions on the time process (e.g., Markov)". These are the classical fixes for the join artefact. — [Nordman 2009](https://arxiv.org/abs/0903.0474)
- Block bootstraps are "designed to keep short-range dependence". The only head-to-head comparison found between a deep generator and a block bootstrap is a GAN "generative bootstrap" against the circular block bootstrap on AR(1). It reports better confidence-interval coverage for the GAN, and was about inference, not long-range fidelity. — [Time series (re)sampling using GANs, arXiv 2102.00208 / Neural Networks 2022](https://arxiv.org/pdf/2102.00208)
- Block bootstrap is used as a non-parametric scenario generator next to parametric generators in macro forecasting, with no benchmark against deep generators. — [MACROCAST, arXiv 2606.28670](https://arxiv.org/pdf/2606.28670). A Symmetric Block Bootstrap has been proposed for synthetic energy-model inputs. — [arXiv 2502.08102](https://arxiv.org/html/2502.08102v1)
- Long et al. (2026) assemble fixed diffusion windows but use **no real-window oracle** and **no beyond-window statistic**. — [arXiv HTML 2610.10222](https://arxiv.org/html/2610.10222)

### Inferences
- **The "window sets a ceiling" half is known in substance.** For any statistic driven by the variance of sums or long-lag autocovariance, independent blocks of length ℓ keep only the Bartlett-weighted autocovariances up to lag ℓ−1. The deficit is about G/ℓ and does not shrink as the generated series gets longer. This is the standard block-bootstrap bias result. The thesis should cite it (Künsch 1989; Hall, Horowitz & Jing 1995; Lahiri 1999/2003; Nordman 2009) rather than present the ceiling as a discovery.
- **The "independent-window oracle predicts the trained model" half appears new.** Three things were not found anywhere: (i) using a seasonal real-window block bootstrap as an empirical upper bound for a windowed *deep* generator; (ii) showing that a trained diffusion model's long-scale deficits (storm duration, sewer overflow) match that oracle at every window length; (iii) using the oracle to choose the window length a downstream decision needs ("sewer needs about a one-day horizon").
- The defensible novelty claim: "We import the block-bootstrap ceiling as a model-free diagnostic for windowed generators. It predicts, before training, which decision-relevant statistics a given window length can and cannot reproduce."
- Caveat for the paper: the −G/ℓ result is about *linear* statistics (variance of sums). Storm duration, run lengths and sewer overflow (thresholded, nonlinear) are not covered by that formula. The thesis's empirical oracle covers them, which is another reason the oracle adds something beyond the classical theory.

**Closeness ratings**
| Work | (a) window-ceiling / oracle | (b) decision-aware |
|---|---|---|
| Nordman 2009 / Lahiri 1999 bias −G/ℓ | PARTIAL (strong): exactly the ceiling for variance of sums under independent blocks, but as a bias of a variance *estimator*, not as a bound on a generator. | DIFFERENT. |
| GAN vs block bootstrap (arXiv 2102.00208) | DIFFERENT: compares coverage and does not use the bootstrap as an upper bound. | DIFFERENT. |
| MACROCAST / Symmetric Block Bootstrap | DIFFERENT: bootstrap as one generator among several, no bound. | DIFFERENT. |
| Long et al. 2026 | PARTIAL: assembly of diffusion windows, no oracle (see Q1). | PARTIAL: downstream utility (see Q1). |

### Gaps
- **No source found that uses a block bootstrap of real windows as an oracle or upper bound for any windowed generator** (searched ML and generic "block bootstrap vs deep generative" queries). This is absence of evidence from a limited search (2 broad searches plus targeted reads), not proof of absence.
- I did not get the original Künsch (1989) or Lahiri (1999) text in this session. The bias formula is taken from Nordman's restatement.
- No result found on block-bootstrap bias for *nonlinear threshold or run-length statistics* (e.g. distribution of spell lengths, exceedance volumes) as a function of ℓ. Lahiri's book might cover functionals of the empirical distribution, but this was not checked.

---

## Q4. Block bootstrap: Künsch, Politis & Romano, Hall-Horowitz-Jing, Politis & White, Lahiri, and hydrological seasonal or matched block bootstraps. What is known about block length versus the statistic of interest?

### Takeaway
The optimal block length grows with sample size at a rate that **depends on the statistic**: n^{1/3} for bias or variance, n^{1/4} for one-sided distribution functions, n^{1/5} for two-sided ones (Hall, Horowitz & Jing 1995). Automatic rules (Politis & White 2004, corrected 2009) estimate it from the spectral density near zero. Hydrology has used the moving-block bootstrap as a streamflow generator since Vogel & Shallcross (1996). It found that block length matters, and that short-memory resamplers lose interannual dependence and drought or run characteristics unless something is added: a parametric pre-whitening step (Srinivas & Srinivasan) or an aggregate-flow conditioning variable (Sharma & O'Neill 2002).

### Cited Findings
**Foundations**
- Künsch (1989), "The jackknife and the bootstrap for general stationary observations", *Ann. Statist.* 17(3):1217-1241: the moving block bootstrap. — [DOI 10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265) [not re-opened; also cited as the MBB source in [Nordman 2009](https://arxiv.org/abs/0903.0474)]
- Carlstein (1986), non-overlapping blocks / subseries variance estimation, *Ann. Statist.* 14(3). — [DOI 10.1214/aos/1176350057](https://doi.org/10.1214/aos/1176350057) [not re-opened; cited as the NBB source in [Nordman 2009](https://arxiv.org/abs/0903.0474)]
- Politis & Romano (1994), "The stationary bootstrap", *JASA* 89(428): random geometric block lengths. — [DOI 10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870) [not re-opened; cited as the SB source in [Nordman 2009](https://arxiv.org/abs/0903.0474)]
- **Hall, Horowitz & Jing (1995)**, "On blocking rules for the bootstrap with dependent data", *Biometrika* 82:561-574. The optimal block length is ∝ n^r, with **r = 1/3 for bias or variance, 1/4 for a one-sided distribution function, 1/5 for a symmetric two-sided distribution function**, for overlapping or non-overlapping blocks. They propose an empirical subsampling and cross-validation rule. — [Semantic Scholar record](https://www.semanticscholar.org/paper/On-blocking-rules-for-the-bootstrap-with-dependent-Hall-Horowitz/3f42f26cd9fb76af48ad33a02c9336172f2049c7); exponents as restated in [Horowitz, "Bootstrap Methods in Econometrics", arXiv 1809.04016](https://arxiv.org/pdf/1809.04016); DOI 10.1093/biomet/82.3.561 [not re-opened]
- The HHJ cross-validation rule is consistent, but "no information on its rate of convergence (besides consistency) has yet been established". It is implemented in the R package `blocklength`. — [Lahiri et al. plug-in rule (ResearchGate record)](https://www.researchgate.net/publication/244964196_A_nonparametric_plug-in_rule_for_selecting_optimal_block_lengths_for_block_bootstrap_methods); [blocklength::hhj docs](https://search.r-project.org/CRAN/refmans/blocklength/html/hhj.html)
- **Politis & White (2004)**, "Automatic block-length selection for the dependent bootstrap", *Econometric Reviews* 23(1). A spectral plug-in rule for SB and CBB, corrected by Patton, Politis & White (2009). — [author PDF](https://mathweb.ucsd.edu/~politis/SBblock-revER.pdf) (the PDF did not load in this session); DOI 10.1081/ETC-120028836 and correction DOI 10.1080/07474930802459016 [not re-opened]; implemented with HHJ in [R package blocklength](https://github.com/Alec-Stashevsky/blocklength)
- Lahiri (1999) *Ann. Statist.* 27:386-404 compares MBB, NBB, CBB and SB, which share the same first-order bias. — cited in [Nordman 2009](https://arxiv.org/abs/0903.0474). Lahiri's monograph *Resampling Methods for Dependent Data* (Springer 2003) is the standard reference. — [Springer DOI 10.1007/978-1-4757-3803-2](https://doi.org/10.1007/978-1-4757-3803-2) [not re-opened]
- In econometrics, Berkowitz & Kilian found parametric methods often more accurate and stressed that "the block size plays an important role in determining the success of the block bootstrap". — [FEDS 1996-45](https://www.federalreserve.gov/econres/feds/recent-developments-in-bootstrapping-time-series.htm)

**Hydrology**
- **Vogel & Shallcross (1996)**, "The moving blocks bootstrap versus parametric time series models", *WRR* 32. MBB "resamples the observed time series using approximately independent moving blocks" and needs only a block-length estimate. In a Monte Carlo study of **reservoir storage capacity**, "the bootstrap always produced storage estimates with lower root-mean-square-error than a parametric alternative". — [DOI 10.1029/96WR00928](https://doi.org/10.1029/96WR00928) (abstract via [EarthRef](https://earthref.org/ERR/68923))
- **Srinivas & Srinivasan (2005)**, "Hybrid moving block bootstrap for stochastic simulation of multi-site multi-season streamflows", *J. Hydrology* 302:307-330. Partial pre-whitening with a parsimonious periodic linear model, MBB of residuals "with an appropriate block size", then post-blackening. It is judged on **run characteristics**. The year is listed as 2004 or 2005 depending on source. — [IITM record](https://publications.iitm.ac.in/publication/hybrid-moving-block-bootstrap-for-stochastic-simulation); [IISc eprint](https://eprints.iisc.ac.in/27611)
- Srinivas & Srinivasan (2006) replace the MBB with the **matched-block bootstrap** and report improved streamflow simulation. — [DOI 10.1016/j.jhydrol.2006.01.023](https://doi.org/10.1016/j.jhydrol.2006.01.023); [IITM record](https://publications.iitm.ac.in/publication/hybrid-matched-block-bootstrap-for-stochastic-simulation)
- **Sharma & O'Neill (2002)**, "A nonparametric approach for representing interannual dependence in monthly streamflow sequences", *WRR* 38(7). Existing monthly generators have "weaknesses in preserving dependence at seasonal to interannual lags". The fix is prior flows for short-term memory plus **an aggregate flow variable for interannual dependence**, and a variable kernel for data with many **zero-flow values**. — [DOI 10.1029/2001WR000953](https://doi.org/10.1029/2001WR000953) (abstract via [EarthRef](https://earthref.org/ERR/75517))
- **Steinschneider et al. (2019)** built a weather-regime-based stochastic weather generator for climate vulnerability assessment of water systems. It uses block bootstrapping with blocks "usually from 1 to 14 days, although it can extend longer than 1 month due to the persistence of WRs". The block bootstrap "will preserve many of the properties of the marginal and joint distributions… but at the expense of being able to simulate values outside the range of the instrumental record". **Precipitation spell statistics were negatively biased by 1-2 days.** — [WRR, DOI 10.1029/2018WR024446](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR024446)
- The nonparametric k-NN bootstrap lineage: Lall & Sharma (1996), [DOI 10.1029/95WR02966](https://doi.org/10.1029/95WR02966); Rajagopalan & Lall (1999) k-NN weather generator, [DOI 10.1029/1999WR900028](https://doi.org/10.1029/1999WR900028). [both not re-opened]

### Inferences
- **The statistic sets the block length.** This is the classical counterpart of the thesis's "sewer needs about a one-day horizon while 80-160 min context suffices for local statistics". HHJ show the optimal ℓ differs by target (variance versus distribution function). Hydrology shows that storage, drought and run statistics need long memory (Vogel & Shallcross; Sharma & O'Neill). The thesis can present "choose the window by the decision statistic" as the generative-model version of this result.
- Steinschneider et al. 2019 is the closest hydrological evidence for the oracle claim. A block-bootstrap generator with blocks up to about 2 weeks *still* underestimates spell lengths by 1-2 days, so a bootstrap underestimates run lengths at the block scale. That is the same symptom the thesis's oracle shows (too few long storms).
- Sharma & O'Neill's aggregate-flow conditioning is the precedent for the thesis's **two-resolution generator** (a coarse aggregate carrying long memory, fine-scale resampling or generation conditioned on it), and should be cited.
- The usual matched-block or tapered fixes reduce *join* bias. They cannot restore dependence longer than the block. Crossfading or bridging diffusion windows therefore cannot be expected to fix long-scale statistics, consistent with the thesis's results and with Long et al.'s finding that continuity does not translate to utility.

**Closeness ratings**
| Work | (a) window-ceiling / oracle | (b) decision-aware |
|---|---|---|
| Künsch 1989 / Carlstein 1986 / Politis-Romano 1994 | PARTIAL: define the independent-block construction whose deficit is the ceiling. | DIFFERENT. |
| Hall, Horowitz & Jing 1995 | PARTIAL: block length must depend on the target statistic; nearest classical form of "the window needed depends on the statistic". | DIFFERENT (statistic-dependence, not decision-dependence; the closest classical idea to "tune to what you use"). |
| Politis & White 2004 (+2009) | PARTIAL: automatic ℓ from the spectral density near zero, i.e. from long-range dependence strength. | DIFFERENT. |
| Lahiri 1999/2003; Nordman 2009 | PARTIAL (strong): the −G/ℓ bias formalises the ceiling for sums. | DIFFERENT. |
| Vogel & Shallcross 1996 | PARTIAL: MBB as a generator, block length matters for a storage-capacity statistic. | PARTIAL: generators judged by a design or decision quantity (reservoir storage), not general fidelity. |
| Srinivas & Srinivasan 2005/2006 | PARTIAL: block resampling of residuals plus parametric memory to recover run characteristics. | DIFFERENT (run and storage statistics, but no decision-maker framing). |
| Sharma & O'Neill 2002 | PARTIAL: names the loss of interannual dependence in short-memory generators and fixes it with aggregate conditioning; handles zero flows. | DIFFERENT. |
| Steinschneider et al. 2019 | PARTIAL: block-bootstrap weather generator underestimates spell length by 1-2 days. | PARTIAL: built for climate-vulnerability (decision) assessment of water systems, but evaluated statistically. |

### Gaps
- The Politis & White PDF did not load (certificate error), and the 2009 correction text was not read. Formulas are not quoted from them.
- I could not confirm Vogel & Shallcross's recommended block length or how storage error changed with ℓ (abstract only).
- No source found tying block length specifically to **sub-daily** rainfall run lengths or to urban drainage or sewer statistics.
- Known theory that the MBB fails or needs modification under **long-range dependence** (e.g. Lahiri 1993) was not verified this session, so it is not cited.
- Vogel & Shallcross and Sharma & O'Neill are cited from their abstracts only.

---

## Q5. Long-range structure in stochastic weather generators: overdispersion and low-frequency variability (brief)

### Takeaway
Daily stochastic weather generators have long been known to **underestimate the variance of monthly and seasonal totals** ("overdispersion", Katz & Parlange 1998). Higher-order occurrence chains remove part of the gap, and the remainder is attributed to low-frequency variation. This is the weather-generator analogue of the window ceiling: a model whose memory is shorter than the aggregation scale under-disperses aggregates.

### Cited Findings
- Katz & Parlange (1998), "Overdispersion phenomenon in stochastic modeling of precipitation": "simple stochastic models of daily precipitation amount tend to underestimate the observed variance of monthly or seasonal total precipitation". For January at Chico, California, more complex chain-dependent models "greatly reduce" the underestimation. The wet-day-count component "can be completely eliminated through the use of a higher-order Markov chain". What remains "could possibly be attributed to low-frequency variation (sometimes termed 'potential predictability')". — [EPFL Infoscience record](https://infoscience.epfl.ch/record/53888). The journal (*J. Climate* 11:591-601) is from prior knowledge and was **not confirmed by the search**.
- Stochastic precipitation generators "often underestimate the interannual variability of precipitation" (citing Katz & Parlange 1998). — [Seasonal hidden Markov model paper, arXiv 1710.08112](https://arxiv.org/pdf/1710.08112)
- A GLM approach to **reducing overdispersion** in weather generators (Kim, Katz et al.), which adds covariates for low-frequency variability. — [KNU research record](https://research.knu.ac.kr/en/publications/reducing-overdispersion-in-stochastic-weather-generators-using-a-/); [NCAR data commons](https://ckanprod.data-commons.k8s.ucar.edu/dataset/77176b91-8298-4001-9b19-44511d2313c3)
- In a regionalised continuous rainfall simulation, "the standard deviation of both wet days and amounts is typically undersimulated at all locations", traced mainly to the daily generator. — [ResearchGate record](https://www.researchgate.net/publication/258724292_Continuous_rainfall_simulation_2_A_regionalized_daily_rainfall_generation_approach)

### Inferences
- The thesis's "too few long storms / too little sewer overflow" result in windowed diffusion is a sub-daily, deep-learning version of overdispersion: a memory horizon shorter than the decision-relevant aggregation scale. Citing Katz & Parlange anchors the window-ceiling point in the hydrometeorology tradition and makes it clear the thesis does not claim to have discovered it.
- The standard weather-generator fixes are higher-order chains (longer memory) and conditioning on low-frequency covariates. They correspond to the thesis's one-day window and two-resolution generator.

**Closeness ratings**
| Work | (a) window-ceiling / oracle | (b) decision-aware |
|---|---|---|
| Katz & Parlange 1998 | PARTIAL: same symptom and cause (short memory → under-dispersed aggregates); no oracle. | DIFFERENT. |
| Kim, Katz et al. (GLM overdispersion) | PARTIAL: fixes the ceiling with low-frequency covariates. | DIFFERENT. |
| Seasonal HMM / continuous-rainfall papers | PARTIAL (symptom only). | DIFFERENT. |

### Gaps
- Katz & Parlange's journal, volume and DOI were not confirmed by the search (EPFL record only).
- The Kim et al. GLM paper's year and venue (believed to be *Climate Research*, 2012) were not confirmed.
- I did not find a weather-generator study that ties overdispersion explicitly to *sub-daily* generators or to urban drainage.
- Wilks & Wilby (1999) "The weather generation game" review, which is likely worth citing for general context, was not verified this session.
