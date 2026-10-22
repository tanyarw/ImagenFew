# Problem-driven (decision-based) scenario generation and reduction in OR and energy systems, and its use with learned generative models

Scope note: this file covers operations research (OR), stochastic programming, energy systems, and the closest neighbouring work in decision-focused learning and model-based RL. It was compiled on 2026-10-22. Each work gets a closeness rating against the proposed headline: *"a scenario generator only needs to be accurate where the decision-makers (controllers/policies) disagree"*. The headline would be built as a decision-weighted fidelity score: generator error weighted by the difference in outcomes between policies. The score should predict whether generated scenarios rank policies as reality does, and standard generative metrics should fail to. The paper would add a fix: select or train the generator to be accurate in that region. The ratings are:
- **SAME IDEA**: the paper already makes this claim.
- **PARTIAL**: it shares a load-bearing component.
- **DIFFERENT**: related background only.

No work found in this search rates SAME IDEA.

## Q1. What do classical problem-driven scenario generation/reduction papers formalise, and is their distance a weighting of distribution error by decision sensitivity?

### Takeaway
Since about 2015 the OR literature has explicitly replaced "distribution-fit" distances with problem-based ones. These come in four forms:
- a supremum over all feasible decisions of the error in expected cost (Henrion & Römisch);
- a "risk region" outside which the distribution can be collapsed (Fairbrother, Turner & Wallace);
- losses computed on a pool of candidate solutions (Prochazka & Wallace; Keutchayan et al.; Narum et al.);
- opportunity-cost matrices (Hewitt et al.; Keutchayan et al.).

All of them weight distribution error by its effect on decision cost, in some form. None of them weights it by the *difference* between two or more policies' outcomes, and all target solving one stochastic program, not ranking a set of policies.

### Cited Findings

**Foundational distance-based and problem-based theory**
- **Henrion & Römisch (2022), "Problem-based optimal scenario generation and reduction in stochastic programming"** (Math. Programming 191:183–205; online 2018; DOI 10.1007/s10107-018-1337-6).
  - They argue that the classical approach (probability metrics such as Fortet–Mourier or Wasserstein) gives only coarse estimates. They propose a "minimal information" semi-distance built only from the model's own cost functions: d_F(P,Q) = sup_{f∈F} |∫ f(ξ)(P−Q)(dξ)|, with F = {f_j(x,·): x ∈ X}. In words, this is the largest error in expected cost over all feasible decisions x.
  - Proposition 1 gives |v(P) − v(Q)| ≤ L·d_F(P,Q), plus a bound on the solution-set deviation. For linear two-stage programs, optimal scenario generation becomes a best uniform approximation of the expected recourse function, posed as a generalised semi-infinite program that is convex when only the right-hand sides or only the costs are random.
  - Sources: [author PDF](https://www2.mathematik.hu-berlin.de/~romisch/papers/HeRo22MP.pdf); [TIB open-access record](https://oa.tib.eu/renate/items/1b9bbb5a-1861-409a-beb6-1923f0340dc4/full); [DOI via Unpaywall](https://unpaywall.org/10.1007%2FS10107-018-1337-6). The quoted definitions were read from the extracted PDF text.
  - **Rating: PARTIAL.** It defines a decision-aware integral probability metric whose function class is "the cost under every decision". But it is a worst-case over all decisions of the *absolute* expected-cost error. It does not use pairwise *differences* between policies, and it does not target ranking.
- **Fairbrother, Turner & Wallace (2022), "Problem-driven scenario generation: an analytical approach for stochastic programs with tail risk measure"** (Math. Programming Ser. B 191(1):141–182; online Nov 2019; arXiv 1511.03074).
  - Key observation: "the value of a tail risk measure depends only on scenarios confined to an area of the distribution that we call the risk region". So all scenarios outside it "can be aggregated into a single point".
  - Definition 3.7: the β-risk region for decision x is R_x(β) = {ξ : F_x(f(x,ξ)) ≥ β}. The risk region for the whole feasible set is the union over x ∈ X.
  - The resulting algorithm, "aggregation sampling", draws samples until enough land in the risk region and aggregates the rest. It is consistent with Monte Carlo sampling. On portfolio problems it gave better and more stable solutions than Monte Carlo.
  - Sources: [arXiv 1511.03074](https://arxiv.org/abs/1511.03074); [NHH open-access PDF](https://openaccess.nhh.no/nhh-xmlui/bitstream/11250/2635391/2/Fairbrother_Turner_Wallace.pdf); [Lancaster record](https://www.research.lancs.ac.uk/portal/en/publications/problemdriven-scenario-generation(f62cf21f-e600-4f60-bd8a-cb52da08b982)/export.html). The definitions were read from the extracted arXiv PDF text.
  - **Rating: PARTIAL.** This is the clearest formal statement that "only part of the distribution matters to the decision" and that the rest can be crude. But the region is defined by the tail of one risk measure across all decisions, not by where policies disagree. The method is analytic, not a learned generator.
- **Arpón, Homem-de-Mello & Pagnoncelli (2018), "Scenario reduction for stochastic programs with Conditional Value-at-Risk"** (Math. Programming 170:327–356; DOI 10.1007/s10107-018-1298-9).
  - Most reduction methods were built for risk-neutral problems and miss the high-cost scenarios that drive CVaR. The authors build a reduction on "effective scenarios" for CVaR objectives.
  - Sources: [Springer DOI](https://link.springer.com/doi/10.1007/s10107-018-1298-9); [Optimization Online preprint](https://optimization-online.org/2018/03/6520/).
  - **Rating: PARTIAL.** It shows that tail and high-cost scenarios carry the decision-relevant information, which is the OR analogue of tail reweighting. It does not use policy disagreement or learned generators.

**Losses computed on candidate solutions (the closest classical analogue of "a set of policies")**
- **Prochazka & Wallace (2020), "Scenario tree construction driven by heuristic solutions of the optimization problem"** (Computational Management Science 17(2):277–307; DOI 10.1007/s10287-020-00369-2).
  - The tree is "driven purely by the out-of-sample performance of a pool of solutions". The loss function compares each solution's out-of-sample performance with its in-tree performance, and minimising it for a fixed number of scenarios gives a problem-specific approximation of the distribution. The loss is usually non-linear and non-convex.
  - The method suits problems that can only be solved for a few scenarios but where out-of-sample evaluation is fast. It also works for binary distributions, where distribution fitting fails.
  - Sources: [IDEAS/RePEc abstract](https://ideas.repec.org/a/spr/comgts/v17y2020i2d10.1007_s10287-020-00369-2.html); [SNF listing](https://snf.no/en/publications/2020/scenario-tree-construction-driven-by-heuristic-solutions-of-the-optimization-problem/). The full text is paywalled and was not read, so the exact loss form, and whether it uses differences or ranks, is unverified.
  - **Rating: PARTIAL, and the closest classical OR work.** It judges a scenario set by how well it reproduces the true performance of a *pool of fixed candidate solutions*, which is structurally like evaluating a set of policies. But the target appears to be per-solution value error, not pairwise differences or ranking, the purpose is building a tree to solve one stochastic program, and no learned generator is involved.
- **Narum, Fairbrother & Wallace (2024), "Problem-based scenario generation by decomposing output distributions"** (EJOR 318(1):154–166; DOI 10.1016/j.ejor.2024.04.006).
  - The method studies "how an output distribution may change across decisions". It evaluates the recourse function over a large discrete distribution for a set of candidate decisions. It then uses singular value decomposition to find a few components that explain most of the variation across those output distributions, and uses them directly as scenarios. It beat distribution-based methods on four case studies.
  - Sources: [IDEAS/RePEc abstract](https://ideas.repec.org/a/eee/ejores/v318y2024i1p154-166.html); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0377221724002650); [NHH news](https://nhh.no/en/departments/business-and-management-science/news/2024/top-publication-by-narum-and-wallace/). The NHH staff page gives pages 327–340, which conflicts with 154–166 everywhere else; 154–166 is more consistent.
  - **Rating: PARTIAL, and conceptually the nearest to "accurate where decisions differ".** It keeps the directions of uncertainty that make outputs vary *across decisions*. But it builds scenarios for one two-stage program, does not evaluate a learned generator, and does not target the ranking of policies.
- **Keutchayan, Ortmann & Rei (2023), "Problem-driven scenario clustering in stochastic optimization"** (Computational Management Science 20(1); DOI 10.1007/s10287-023-00446-2; arXiv 2106.11717).
  - It minimises the *implementation error*, "the error incurred by implementing the solution of the reduced problem in the original problem". Each cluster's discrepancy measures how well the representative scenario's cost matches the cluster's average cost over a set X̃ of candidate solutions.
  - In practice X̃ is the set of single-scenario optimal decisions x*_i, which gives the opportunity-cost matrix V_ij = F(x*_i, ξ_j). The authors say a valid X̃ "would have to include the optimal solutions of the original and approximate problems" or any "sensible" solutions.
  - Sources: [arXiv 2106.11717](https://arxiv.org/abs/2106.11717); [IDEAS](https://ideas.repec.org/a/spr/comgts/v20y2023i1d10.1007_s10287-023-00446-2.html). The definitions were read from the extracted arXiv PDF text.
  - **Rating: PARTIAL.** Scenario error is measured through the cost of a *set of candidate decisions*, which is close to "a set of policies". But it measures the value error of those decisions, not their pairwise differences, and the goal is reduction for one program.
- **Hewitt, Ortmann & Rei, "Decision-based scenario clustering for decision-making under uncertainty"** (Annals of OR 315(2), 2022; online 2021; DOI 10.1007/s10479-020-03843-x).
  - It groups scenarios by the decisions associated with them, using a graph built on "the opportunity cost of predicting the wrong scenario".
  - Sources: [IDEAS](https://ideas.repec.org/a/spr/annopr/v315y2022i2d10.1007_s10479-020-03843-x.html); [CIRRELT preprint](https://www.cirrelt.ca/documentstravail/cirrelt-2018-39.pdf).
  - **Rating: PARTIAL.** It is decision-based similarity: two scenarios are "the same" if they induce the same decisions. That is the dual of "the generator may err freely where decisions do not change".
- **Bertsimas & Mundru (2023), "Optimization-based scenario reduction for data-driven two-stage stochastic optimization"** (Operations Research 71(4):1343–1361; DOI 10.1287/opre.2022.2265).
  - A convex optimisation-based reduction that "takes into account the objective and problem structure". It recovers the SAA (sample average approximation) solution as the number of scenarios grows. Keeping about 1–2% of n scenarios, it beats importance sampling, Monte Carlo and squared-Euclidean Wasserstein reduction.
  - Sources: [Optimization Online](https://optimization-online.org/2022/01/8773/); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/144108); [IDEAS](https://ideas.repec.org/a/inm/oropre/v71y2023i4p1343-1361.html). The exact problem-dependent distance was not visible in the abstract pages and is unverified here.
  - **Rating: PARTIAL.** It is objective-aware reduction for one program, with no policy comparison and no learned generator.

**Power-systems problem-driven reduction**
- **Feng & Ryan, solution-sensitivity-based scenario reduction for stochastic unit commitment** (Computational Management Science, 2014/2016).
  - The heuristic "forward selection in recourse clusters" (FSRC) picks scenarios by their *cost and reliability impact*, not by probability distance. On ISO-derived instances it gave more reliable commitments at similar cost than probability-metric reduction.
  - Sources: [Iowa State repository](https://lib.dr.iastate.edu/imse_pubs/12) (redirected; returned 403 when fetched); [Feng dissertation](https://www.imse.iastate.edu/files/2014/03/Dissertation-Yonghan-Feng.pdf). The journal DOI and volume could not be verified.
  - **Rating: PARTIAL.** It is decision-impact-based selection in energy, for one unit commitment model.
- **Zhuang, Cheng, Qi, Almassalkhi & Liu, "An iterative problem-driven scenario reduction framework for stochastic optimization with CVaR"** (arXiv 2510.15251, Oct 2025; cited elsewhere as PSCC 2026, but that venue is secondary-source only).
  - It maps scenarios "from distribution space into problem space" and reports an optimality gap below 1%. It argues that distribution-driven methods "often exclude extreme scenarios".
  - Precursor: Zhuang et al., "Problem-driven scenario reduction framework for power system stochastic operation", arXiv 2404.07810.
  - Sources: [arXiv 2510.15251](https://arxiv.org/abs/2510.15251); [arXiv 2404.07810](https://arxiv.org/pdf/2404.07810).
  - **Rating: PARTIAL.** It is the energy-systems version of problem-driven, tail-aware reduction, for one decision problem.

**Evaluating scenario generators by decisions**
- **Kaut & Wallace (2007), "Evaluation of scenario-generation methods for stochastic programming"** (Pacific Journal of Optimization; 2003 SPEPS e-print).
  - It sets minimal requirements and the standard in-sample and out-of-sample *stability* tests for judging a scenario generator by the decisions and objective values it yields, rather than by distribution fit. It notes that no method is best for all models.
  - Sources: [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/45403); [HU Berlin e-print](https://edoc.hu-berlin.de/18452/8948).
  - **Rating: PARTIAL.** It established "judge a scenario generator by decision quality". The evaluation is per single program, not policy ranking, and there is no error weighting.

### Inferences
- In Henrion & Römisch's framework, the headline score corresponds to choosing the function class F_Δ = {f_a − f_b : a, b policies}. Generator error measured in this class is |∫(f_a − f_b) d(P−Q)| ≤ ∫ |f_a − f_b| d|P−Q|, and the right-hand side is exactly "generator error weighted by policy disagreement". (This is my own derivation, not a statement from the papers.) A reviewer from stochastic programming is likely to say the headline is "d_F with a different F". The paper should cite Henrion & Römisch and position the contribution as follows:
  - choosing the *policy-difference* class;
  - targeting *ranking* rather than value;
  - closed-loop sequential controllers rather than two-stage programs;
  - evaluating and selecting *learned* generators.
- The "pool of candidate solutions" device (Prochazka & Wallace; Keutchayan et al.; Narum et al.) is the classical OR precedent for evaluating scenarios against a finite set of decisions. The thesis's set of RL and rule-based controllers plays the same role, so these three papers are must-cites.
- Fairbrother et al.'s risk region (the union over decisions of tail regions) is the formal precedent for "only part of the distribution matters". The thesis's region (where the controllers' outcomes differ) is a different and generally smaller set: tails where every policy fails equally are excluded.

### Gaps
- I could not read the full text of Prochazka & Wallace (2020), so I could not verify whether their loss uses absolute value errors or rank/difference errors. This is the single most important paper to read in full before claiming novelty.
- The exact form of Bertsimas & Mundru's problem-dependent distance was not verified.
- Feng & Ryan's journal DOI could not be confirmed (the repository returned 403).
- The NHH page numbers for Narum et al. conflict (327–340 vs 154–166).

## Q2. Decision-focused / task-based scenario generation with learned (deep) generators, 2017–2026

### Takeaway
Since 2024–2026 a fast-growing literature trains deep generators (VAE, GAN, normalizing flow, diffusion) against a downstream optimisation loss. Examples: Diff2SP, Diffusion-DFL, Gen-DFL, decision-focused grid-dispatch scenario generation, decision-focused contextual scenario generation, and control-oriented scenario trees. All of them optimise or evaluate the generator for *one* decision problem or controller. None defines fidelity as error weighted by *inter-policy* outcome differences, and none uses the generator to *rank* policies. Their bounds link regret to plain Wasserstein distance, not to a decision-weighted distance.

### Cited Findings
**Foundations**
- **Donti, Amos & Kolter (NeurIPS 2017), "Task-based end-to-end model learning in stochastic optimization"** (arXiv 1703.04529).
  - It trains *probabilistic* models end-to-end against the downstream stochastic-programming objective instead of likelihood. Tasks: inventory, grid scheduling and battery arbitrage. It reports a 38.6% improvement over a tuned stochastic programming scheduler.
  - Sources: [arXiv](https://arxiv.org/abs/1703.04529); [code](https://github.com/locuslab/e2e-model-learning).
  - **Rating: DIFFERENT.** It is the root of the "task loss for a distribution model" idea, for one policy class and with no ranking.
- **Grigas, Qi & Shen, "Integrated Conditional Estimation-Optimization"** (Operations Research, DOI 10.1287/opre.2023.0427; arXiv 2110.12351).
  - It fits a conditional *distribution* with an objective aligned to the downstream optimisation. It gives consistency and finite-sample generalisation bounds and handles model mismatch.
  - Sources: [arXiv](https://arxiv.org/abs/2110.12351v3); [INFORMS](https://pubsonline.informs.org/doi/epdf/10.1287/opre.2023.0427).
  - **Rating: DIFFERENT.** It is decision-aware density estimation for one decision.
- **Mandi et al. (JAIR 2024), "Decision-focused learning: foundations, state of the art, benchmark and future opportunities"** (JAIR 80:1623–1701; DOI 10.1613/JAIR.1.15320; arXiv 2307.13565).
  - It reviews gradient-based and gradient-free DFL methods and benchmarks 11 methods on 7 problems.
  - Sources: [arXiv](https://arxiv.org/abs/2307.13565v4); [ML Anthology](https://mlanthology.org/jair/2024/mandi2024jair-decisionfocused).
  - **Rating: DIFFERENT.** This is the survey to cite for DFL.
- **Mandi, Bucarey, Mulamba & Guns (ICML 2022), "Decision-focused learning: through the lens of learning to rank"** (PMLR 162:14935–14947; arXiv 2112.03609).
  - It recasts DFL as learning an objective that *ranks feasible solutions correctly*, using pointwise, pairwise and listwise losses over a subset of solutions.
  - Sources: [PMLR](https://proceedings.mlr.press/v162/mandi22a.html); [arXiv](https://arxiv.org/abs/2112.03609v4).
  - **Rating: PARTIAL.** It states explicitly that a model only needs to get the *ordering of candidate solutions* right, with a pairwise loss over solution pairs. But it applies to point predictors of cost coefficients, not to scenario generators evaluating closed-loop policies.

**Learned generators trained with downstream loss**
- **Sun & Liu (arXiv 2606.05649, June 2026), "Diff2SP: Diffusion models for correlated scenario generation in stochastic programming"**.
  - Each diffusion step minimises λ_noise·(noise MSE) + λ_recon·(reconstruction MSE) + λ_opt·(squared gap between the optimal values under true and generated samples).
  - Lemma 1 bounds regret by R(θ) ≤ 2L·W₁(P, Q_θ). This is plain 1-Wasserstein scaled by the Lipschitz constant of the cost, so it is *not* decision-weighted. Remark 6 says the regret analysis needs the optimisation loss to get "sufficient weight".
  - They compare GAN, vanilla diffusion and Diff2SP on a quadratic program and DC-OPF. They hold the first-stage decision fixed and compare its objective under true and generated scenarios. They do not rank multiple policies.
  - They name decision-focused generative learning (Wang et al. 2025, i.e. Gen-DFL) as the closest prior work.
  - Source: [arXiv HTML](https://arxiv.org/html/2606.05649v1).
  - **Rating: PARTIAL.** It is a diffusion scenario generator trained with a decision term and is the nearest learned-generator analogue. Its theory uses the uniform Lipschitz–Wasserstein bound that a decision-weighted score would tighten. It has one decision problem and no policy ranking.
- **Zhao, Yeh, Kong & Wang (arXiv 2510.11590, Oct 2025), "Diffusion-DFL: Decision-focused diffusion models for stochastic optimization"**.
  - It is described as the first diffusion-based DFL method. Gradients come either by reparameterisation through sampling (memory-heavy) or from a lightweight weighted score-function estimator.
  - Sources: [arXiv](https://arxiv.org/abs/2510.11590).
  - **Rating: DIFFERENT.** It is a diffusion generator trained for one downstream decision.
- **Wang, Chen, Liang, Fioretto & Zhu (ICLR 2026), "Gen-DFL: Decision-focused generative learning for robust decision making"** (arXiv 2502.05468).
  - It argues that point-prediction DFL struggles in high-dimensional and risk-sensitive settings. It learns a generative model of the parameters and draws "samples from the tail regions of the learned distribution" for robustness, with improved worst-case bounds versus DFL.
  - Sources: [arXiv](https://arxiv.org/abs/2502.05468); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c25c15b5b2fd386ab188a918e54c7d5-Abstract-Conference.html).
  - **Rating: PARTIAL**, for the tail-sampling or reweighting fix. It is not policy-ranking and not an error-weighted fidelity score.
- **Zhou, Zhou, Morstyn & Wang (arXiv 2607.05830, July 2026), "Decision-focused scenario generation and selection for efficient and robust grid dispatch"**.
  - VAE, GAN and diffusion generators are trained on the downstream operational cost they induce in distributionally robust dispatch. A differentiable selector picks decision-relevant scenarios from a generated pool. Operational cost is 0.80–2.02% lower than with accuracy-oriented generators.
  - Sources: [arXiv](https://arxiv.org/abs/2607.05830).
  - **Rating: PARTIAL.** It does decision-aware training *and* selection of learned generators in power systems, which matches the "fix" part. But it serves a single dispatch policy and does not rank policies.
- **Hornewall, Delannoy-Pavy, Leclère & Homem-de-Mello (NeurIPS 2025), "Decision focused scenario generation for contextual two-stage stochastic linear programming"**.
  - A neural generator maps context to a fixed-size scenario set. It is trained end-to-end on the true downstream cost via implicit differentiation of a log-barrier-regularised deterministic equivalent.
  - Sources: [NeurIPS 2025 page](https://neurips.cc/virtual/2025/135747).
  - **Rating: DIFFERENT.** It optimises one contextual decision.
- **Islip, Kwon, Bae & Kim (arXiv 2502.05349, 2025; revised 2026), "Contextual scenario generation for two-stage stochastic programming"**.
  - It has a distributional variant (kernel distance) and a task-based variant that differentiates through a learned surrogate of the second-stage objective, with finite-sample generalisation guarantees.
  - Sources: [arXiv](https://arxiv.org/abs/2502.05349).
  - **Rating: DIFFERENT.** It contrasts distributional and task-based training for one decision.
- **Pavirani, Claessens, Pinson & Develder (arXiv 2608.09335, Aug 2026), "Control-oriented scenario tree construction through reinforcement learning"**.
  - It argues that "better distributional accuracy doesn't guarantee better control". An attention-based RL policy assigns sampled forecasts to tree leaves and is trained on *closed-loop* MPC profit. The learned trees beat forward and backward Wasserstein reduction and certainty-equivalent control on risk-averse battery arbitrage. They "branch selectively around high-impact events" and keep most trajectories nearly deterministic.
  - Sources: [arXiv](https://arxiv.org/abs/2608.09335).
  - **Rating: PARTIAL, and the closest in the *control* setting.** Scenario representation is judged by closed-loop controller outcome and ends up concentrated on high-impact events. But it serves a single controller and does not compare or rank controllers.

**Value-oriented forecasting in energy (point forecasts, not generators)**
- **Zhang, Wen, Bian & Shi (arXiv 2405.09004)**, "Improving sequential market clearing via value-oriented renewable energy forecasting", derives a piecewise-linear value-oriented loss. **Zhang et al. (arXiv 2309.00803)**, "Toward value-oriented renewable energy forecasting: an iterative learning approach". **Wen & Pinson (arXiv 2501.16086)**, "Value-oriented forecast reconciliation for renewables in electricity markets".
  - Sources: [2405.09004](https://arxiv.org/pdf/2405.09004); [2309.00803](https://arxiv.org/pdf/2309.00803); [2501.16086](https://arxiv.org/html/2501.16086v1).
  - **Rating: DIFFERENT.** These are value-oriented *point forecasts* for one market or dispatch decision. They are the "value-oriented" terminology to cite.

**Evaluating learned energy generators by "quality" and "value"**
- **Dumas, Wehenkel et al. (Applied Energy 305, 2022, 117871), "A deep generative model for probabilistic energy forecasting in power systems: normalizing flows"** (arXiv 2106.09370).
  - It compares normalizing flows with GANs and VAEs on GEFCom2014 wind, solar and load. Scenarios are assessed for *quality* (several metrics) and for *value* (an energy retailer case study).
  - A follow-up (arXiv 2212.02977; exact title not verified here) reportedly adds DDPMs, which were better in both quality and value.
  - Sources: [arXiv 2106.09370](https://arxiv.org/abs/2106.09370v3); [arXiv 2212.02977](https://arxiv.org/pdf/2212.02977v4).
  - **Rating: PARTIAL.** It evaluates deep generators by downstream value next to statistical quality. But it uses one decision model, and it does not test whether quality metrics predict value rankings, or weight error by policy disagreement.
- **Chen, Wang, Kirschen & Zhang, "Model-free renewable scenario generation using generative adversarial networks"** (arXiv 1707.09676; IEEE TPWRS 2018 per the Diff2SP citation).
  - It is the standard GAN scenario-generation baseline in power systems, trained on statistical fit only.
  - Sources: [arXiv](https://arxiv.org/pdf/1707.09676v2).
  - **Rating: DIFFERENT.**

### Inferences
- The learned-generator + decision-loss space is crowded in 2025–2026 and clearly published. A thesis paper cannot claim "decision-aware training of a diffusion scenario generator" as new. At minimum it must cite Diff2SP, Diffusion-DFL, Gen-DFL, Zhou et al. 2026 and Pavirani et al. 2026.
- All these works share one decision-maker. The distinctive move in the headline is a *set* of competing policies with *differences* as the target. That setting is not covered by any learned-generator paper found.
- Diff2SP's 2L·W₁ bound is a useful foil. A decision-weighted bound, ∫ |f_a − f_b| d|P−Q|, can be much tighter whenever the policies agree on most of the distribution's mass. This is plausible for sewer control, where policies coincide in dry weather. (Inference, not verified against any paper.)

### Gaps
- I did not verify learned scenario reduction and representation work for two-stage stochastic programs: Wu et al. ICLR 2022 "learning scenario representation"; Bengio et al. 2020, a learned representative scenario; Dumouchelle et al. Neur2SP 2022. These are probably DIFFERENT, since they speed up one stochastic program, but they were not checked.
- The search returned "Diffusion-based scenario tree generation for multivariate time series prediction and multistage stochastic optimization" (arXiv 2509.14832), but its content was not verified.
- The full loss of Zhou et al. 2026 (whether a distribution-fit term is kept) was not visible in the abstract.

## Q3. Has any of this been used to compare or rank several policies/controllers, or to weight generator error by the outcome difference between policies?

### Takeaway
I found no OR or energy paper that weights generator error by inter-policy outcome differences. I also found none that tests whether generated scenarios preserve the ranking of a set of policies against held-out reality, or reports that standard generative metrics fail at that while a decision-weighted score succeeds. The closest precedents are:
- OR losses computed on pools of candidate solutions (Q1);
- learning-to-rank DFL (Mandi et al. 2022);
- policy selection by out-of-sample simulation (Defourny et al.);
- RL model learning in which the model only needs to be correct for a *set of policies* (value equivalence) or with error weighted by value sensitivity (VAML, VaGraM).

The last group is outside OR/energy but is the most dangerous for a novelty claim.

### Cited Findings
**Policy selection with scenario simulators in stochastic programming**
- **Defourny, Ernst & Wehenkel, "Scenario trees and policy selection for multistage stochastic programming using machine learning"** (arXiv 1112.4463; INFORMS JoC version recalled from memory, not verified).
  - They generate many scenario trees, extract policies from their solutions with supervised learning, and *select* the best by out-of-sample simulation on independent scenarios. This ranks scenario trees or derived policies by simulated performance.
  - Sources: [arXiv](https://arxiv.org/pdf/1112.4463).
  - **Rating: PARTIAL.** Policies are ranked by simulation on scenarios, but the scenario generator's own fidelity and how it is weighted are not studied.
- **Kaut & Wallace (2007)**: out-of-sample evaluation, i.e. judging scenario sets by "the performance of their corresponding optimal solutions within the original formulation". Later papers quote this as the standard benchmark practice.
  - Sources: [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/45403); see also the restatement in [Islip et al.](https://arxiv.org/html/2502.05349v2).
  - **Rating: PARTIAL**, as in Q1.

**Do statistical scenario metrics predict decision value? (energy evidence)**
- **Sari, Lee, Ryan & Woodruff (2016), "Statistical metrics for assessing the quality of wind power scenarios for stochastic unit commitment"** (Wind Energy 19:873ff).
  - They propose a mass-transportation-distance rank histogram that best separates scenario sets by bias, variability and autocorrelation. A companion Sari & Ryan report relates the metrics to actual stochastic unit commitment (SUC) cost: lack of uniformity in that histogram can screen out sets that lead to excess no-load or penalty costs.
  - Sources: [bepress listing](https://works.bepress.com/sarah_m_ryan/33); [Sari dissertation](https://www.imse.iastate.edu/files/2014/03/SariDidem-dissertation.pdf).
  - **Rating: DIFFERENT, but it is motivating evidence** that statistical metrics only partly predict decision cost.
- **Staid, Watson, Woodruff & Rachunok** (Sandia/BPA study; FERC technical conference slides).
  - Assessing wind-scenario methods on Bonneville Power Administration (BPA) data, they conclude that "the best metrics cannot tell us much about scenario quality" for stochastic unit commitment.
  - Sources: [FERC slides](https://www.ferc.gov/sites/default/files/2020-08/T2-B1_Staid.pdf); [Sandia publication record](https://www.sandia.gov/research/publications/details/assessment-of-wind-power-scenario-creation-methods-for-stochastic-power-sys-2020-06-15/). These are slides, not a peer-reviewed paper; the quote is from the search summary.
  - **Rating: DIFFERENT.** It is supporting evidence that standard metrics fail to predict downstream quality.
- **Pinson & Girard (2012), "Evaluating the quality of scenarios of short-term wind power generation"** (Applied Energy 96:12–20; DOI 10.1016/j.apenergy.2011.11.004).
  - It argues that marginal-only verification ignores temporal dependence, which matters for multi-stage decisions. It proposes multivariate and event-based diagnostics.
  - Sources: [IDEAS](https://ideas.repec.org/a/eee/appene/v96y2012icp12-20.html); [HAL](https://hal-mines-paristech.archives-ouvertes.fr/hal-00654836).
  - **Rating: DIFFERENT.** It is a statistical-quality framework; the motivation mentions decisions.

**Model-based RL: "accurate only where values/policies need it" (outside OR, but closest conceptually)**
- **Farahmand, Barreto & Nikovski (AISTATS 2017), "Value-aware loss function for model-based reinforcement learning"**.
  - It argues that fitting a generative transition model by log-loss "is an overkill". The model should instead be penalised by the error it induces in value-function backups.
  - Sources: [PMLR v54](https://proceedings.mlr.press/v54/farahmand17a.html).
  - **Rating: PARTIAL.** The model error is weighted by its effect on value. It uses a single value function or a supremum over a function class, not inter-policy differences.
- **Voelcker, Liao, Garg & Farahmand, "Value gradient weighted model-based reinforcement learning" (VaGraM)** (arXiv 2204.01464; ICLR 2022, from memory).
  - It "penalize[s] a model prediction by the resulting difference in a value function" and weights model error by value-function *gradients* to make VAML practical with neural networks.
  - Sources: [arXiv](https://arxiv.org/pdf/2204.01464).
  - **Rating: PARTIAL.** This is a direct mechanism precedent for "generator error × outcome sensitivity". It uses one value function and does not compare policies.
- **Grimm, Barreto, Singh & Silver (NeurIPS 2020), "The value equivalence principle for model-based reinforcement learning"**, plus "Proper value equivalence" (NeurIPS 2021) and "Approximate value equivalence" (NeurIPS 2022).
  - Two models are value-equivalent with respect to a *set of policies* and functions if they give the same Bellman updates. As the set grows, the equivalent models shrink to the true model. Proper VE needs only the set of policies.
  - Sources: [NeurIPS 2020 paper](https://papers.neurips.cc/paper_files/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf); [arXiv 2011.03506](https://arxiv.org/abs/2011.03506v1); [Proper VE arXiv 2106.10316](https://arxiv.org/pdf/2106.10316v2); [Approx. VE NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/d53538ba21c05fa361d2b21704172753-Paper-Conference.pdf).
  - **Rating: PARTIAL, and the strongest conceptual precedent overall.** It formalises that "a model need only be correct with respect to the policies you care about". But it concerns planning or learning in RL, not empirically evaluating learned scenario generators by policy-ranking fidelity, and it defines no outcome-difference weighting.

### Inferences
- The specific combination below appears unpublished as of 2026-10-22, at least in the OR, energy and DFL venues searched:
  1. the target is *ranking of a set of policies/controllers*;
  2. the fidelity score weights generator error by *outcome differences between policies*;
  3. it is validated on *learned deep generators* (about 28 diffusion variants plus classical baselines);
  4. the setting is a closed-loop physics simulator with real held-out years as ground truth;
  5. it is shown to predict ranking agreement where standard generative metrics do not.
- The strongest overlaps a reviewer would raise:
  - Henrion & Römisch's d_F, since the score is an integral probability metric over a decision-derived function class;
  - Prochazka & Wallace and Keutchayan et al., which evaluate scenario sets on pools of candidate solutions;
  - Narum et al., which keeps the variation of outputs across decisions;
  - Grimm et al.'s value equivalence, where the model only needs correctness for a policy set;
  - VAML/VaGraM, where model error is weighted by value sensitivity.
- The paper should state precisely what differs: pairwise *differences*, not values; *ranking* as the target metric (e.g. Kendall τ or pairwise sign agreement); a *post hoc* evaluation score that predicts ranking agreement before any decision is optimised; closed-loop sequential controllers; and empirical validation against held-out real years.
- The "fix" (decision-aware training or selection, tail reweighting) overlaps with Diff2SP, Zhou et al. 2026, Gen-DFL (tail sampling), Fairbrother et al. and Arpón et al. (tail and high-cost focus), and Pavirani et al. (high-impact branching). It is the weaker novelty axis. The evaluation score is the stronger one.

### Gaps
- I did not search the off-policy evaluation (OPE) literature in RL for "rank correlation of policy-value estimates under model misspecification". Examples are model-based OPE benchmarks that report Spearman ρ or regret@k. This is the most likely place for a near-SAME-IDEA result outside OR and should be covered by another researcher.
- I did not search hydrology or water-resources decision-making for "decision scaling" (Brown et al., WRR 2012; venue from memory, unverified) or "scenario discovery". Both ask where in the input space decisions change, and both may be close.
- Full texts of Defourny et al. and Prochazka & Wallace were not read.

## Q4. Known results on when a biased scenario set still ranks decisions correctly

### Takeaway
There are three relevant strands, but none gives a ready-made "decision-weighted" theorem for biased learned generators:
- **Simulation/OR, ordinal optimisation:** ordering converges exponentially fast while values converge at O(1/√n), and crude models can still rank designs well.
- **SAA theory:** the probability of selecting the true optimum among finitely many decisions goes to one exponentially, at a rate set by the optimality gap.
- **Water-resources decision-making under deep uncertainty:** empirical evidence that scenario selection shifts robustness *values* much more than it changes *rankings*, with some counter-evidence.

The common-mode cancellation argument (bias where policies behave alike drops out of differences) is implicit in these results but was not found stated as a theorem for scenario generators.

### Cited Findings
- **Ho, Sreenivas & Vakili (1992), "Ordinal optimization of DEDS"** (Discrete Event Dynamic Systems 2:61–88; DOI 10.1007/BF01797280).
  - It argues for "ordinal rather than cardinal optimization": finding good, better or best designs instead of estimating their performance accurately. It is the foundation of ordinal optimisation.
  - Sources: [DOI](https://www.doi.org/10.1007/BF01797280); [Academia copy](https://www.academia.edu/53100736/Ordinal_optimization_of_DEDS).
  - **Rating: PARTIAL.** It is the canonical "ranking is easier than value" result. The ordinal optimisation literature also says crude analytical or simplified simulation models "may be able to provide good estimates of the ranking". That claim comes from a tutorial-level source and is not a formal theorem about biased models.
  - Supporting sources: [DTIC report](https://apps.dtic.mil/sti/pdfs/ADA350586.pdf); [JORS 2000 review](https://link.springer.com/article/10.1057/palgrave.jors.2600906).
- **Exponential convergence of ordinal comparison.** The probability of correct selection converges exponentially, while point estimates converge at O(1/√n). This is proven for regenerative simulations.
  - Sources: search summary of the [DTIC report](https://apps.dtic.mil/sti/pdfs/ADA350586.pdf) and [Springer article BF02190101](https://link.springer.com/article/10.1007/BF02190101). The second is believed to be Dai (1996, JOTA) on the convergence of ordinal comparison, but it was not opened (the fetch redirected), so treat this attribution as unverified.
  - Caveat: Glynn & Juneja (2015) argue that some guarantees built on this exponential rate are misleading. Source: [arXiv 1507.04564](https://arxiv.org/pdf/1507.04564v1).
- **Kleywegt, Shapiro & Homem-de-Mello, "The sample average approximation method for stochastic discrete optimization"** (journal version believed to be SIAM J. Optim. 2002; not verified here).
  - With a finite decision set, the SAA solution coincides with the true optimum with probability approaching one *exponentially* in the sample size. The constant depends on how well conditioned the true optimal set is, i.e. the gap to the second-best decision.
  - Sources: [HU Berlin e-print](https://edoc.hu-berlin.de/items/ee3b4ebc-63a0-47fe-bed8-58f01c079bf5/full); [WSC 2001 paper](https://informs-sim.org/wsc01papers/056.PDF).
  - **Rating: DIFFERENT.** This is for *unbiased* sampling, but its margin and gap logic is the template for a ranking-preservation condition.
- **McPhail, Maier, Westra, Kwakkel & van der Linden (2020), "Impact of scenario selection on robustness"** (Water Resources Research 56, e2019WR026515; DOI 10.1029/2019WR026515).
  - Scenario selection "has a large influence on the robustness value calculated for each decision alternative" but "a relatively small influence on how those decision alternatives are ranked" (Lake Problem case study).
  - Source: [deepuncertainty.org summary](https://www.deepuncertainty.org/2020/09/11/impact-of-scenario-selection-on-robustness-2020).
  - **Rating: PARTIAL.** It is direct empirical evidence that biased scenario sets can preserve rankings while missing values. It has no generator-fidelity score and no explanation of *when*.
- **Quinn, Hadjimichael, Reed & Steinschneider (2020), "Can exploratory modeling of water scarcity vulnerabilities and robustness be scenario neutral?"** (Earth's Future 8(11); DOI 10.1029/2020EF001650).
  - Different scenario designs (scenario-neutral, projection-, paleo-informed) gave different sensitivities *and robustness rankings*.
  - Source: [ESS Open Archive](https://essopenarchive.org/users/551389/articles/604354-can-exploratory-modeling-of-water-scarcity-vulnerabilities-and-robustness-be-scenario-neutral).
  - **Rating: PARTIAL.** It is counter-evidence: rankings are *not* always preserved, which motivates a score that predicts when they are.
- A TU Delft "Belief-Informed Robust Decision Making" flood study reports that non-uniform sampling distributions "change the robustness and ranking of measures".
  - Source: [TU Delft repository](https://repository.tudelft.nl/file/File_3af857e6-eaa7-4576-a630-b12241b3e705?preview=1).
  - Rating: PARTIAL, as counter-evidence. Author list and year not verified.

### Inferences
- A simple sufficient condition for the generated scenarios Q to rank policies a and b as reality P does is my derivation, not a sourced result. With Δ_ab = f_a − f_b, the ranking is preserved if |∫Δ_ab d(P−Q)| < |E_P[Δ_ab]|. The left side is bounded by ∫ |Δ_ab| d|P−Q|, which is the decision-weighted fidelity.
  - Common-mode bias, meaning error in regions where Δ_ab ≈ 0 (e.g. dry weather, or storms so extreme that every policy overflows), contributes nothing.
  - Errors matter in proportion to policy disagreement, relative to the true performance gap (the margin).
  - This is the scenario-generator analogue of the ordinal optimisation and SAA margin arguments and of McPhail et al.'s empirical finding. Writing it down explicitly and validating it empirically appears to be new, but a reviewer may call it an immediate corollary of integral-probability-metric theory (Henrion & Römisch). So the empirical predictive-validity result (the score predicts ranking agreement across about 28 generator variants, standard metrics do not) is what carries the paper.
- The Fairbrother et al. risk region and the policy-disagreement region differ in an informative way. A tail where every controller overflows is in the risk region but not in the disagreement region. The thesis could show that tail-weighted metrics (such as extreme-value fit) over-penalise generators for errors that do not affect rankings.

### Gaps
- I found no theorem stating that stochastic dominance between two policies' outcome distributions is preserved under a specific class of generator bias (e.g. monotone or common-mode distortions). This was not searched exhaustively.
- I did not verify a citation for common random numbers or variance reduction in paired comparisons (e.g. Mak, Morton & Wood 1999, which compares candidate solutions with common samples). Its relevance is that paired comparisons cancel common-mode noise.
- Dai (1996) and Kleywegt et al. (2002) journal metadata were not confirmed from primary pages.
