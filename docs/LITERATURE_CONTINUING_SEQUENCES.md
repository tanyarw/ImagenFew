# Literature: making long sequences by continuing short pieces

**Written:** 2026-10-01 · **For:** the thesis (related-work section) and deciding the next experiment.
**Related:** `notebooks/research_evaluation.ipynb` §5.3 and §7; `code_plan/NEXT_STEPS_AFTER_v14.md`;
the earlier survey in `docs/ATTENTION_AND_RECEPTIVE_FIELD_AUDIT.md` §4.

Every paper below was checked against its published version or abstract (links at the end).
The "what it means for us" notes are my reading, not the authors' claims.

---

## The question in plain words

Our model makes rain in 5-hour pieces and glues them together. Each piece is made without
seeing the one before, so long storms break at the joins. Our latest attempt gave each new piece
the end of the previous piece as a fixed starting point ("a head start"), without retraining the
model, and used a "redo each step several times" trick to make the new part fit the head start.
Storms then continued across joins, but the rain came out about half as heavy as it should.

Three questions for the literature:
1. Has anyone made long sequences by continuing short pieces, and how?
2. Does the "head start without retraining" shortcut work, and is the redo trick known to cause
   problems?
3. Has anyone done this for rain?

---

## 1. The redo trick comes from photo repair

**RePaint** (Lugmayr et al., CVPR 2022) repairs missing parts of photos with a model that was
only ever trained to make whole photos. At each clean-up step it pastes in the known part of the
photo, and it goes back and redoes steps several times ("resampling") so the new part blends in
with the known part.

**What it means for us.** This is exactly the method our latest attempt used. It was designed and
tested on photos (faces, ImageNet), not on data that is mostly zeros.

**A known weakness.** Rout, Parulekar, Caramanis and Shakkottai (2023) proved that RePaint's
results are **biased**: they match the known part but are pulled away from the right answer in
the part being filled in, even in the simplest possible case, and they proposed a corrected
version (RePaint⁺). The RePaint authors themselves note that it reproduces the imbalances of
the training data (their ImageNet model filled holes with dogs too often, because dogs are
over-represented).

**What it means for us.** Our dry rain fits both observations: the filled-in part drifts towards
what is most common in the training data, and for rain that is "dry" (91% of steps). I found no
paper reporting this specifically for mostly-zero data such as rainfall, so this may be a small
original finding, worth stating carefully as such.

---

## 2. "Just paste in the known part" was tried for video, and did not work well

**Video Diffusion Models** (Ho, Salimans and others, NeurIPS 2022) extended videos by pasting the
previous frames into a model not trained for it (their "replacement method"). They report that the
new frames looked fine on their own but often **did not follow on coherently** from the given
frames, and they introduced a better way of steering the model ("reconstruction guidance").

**What it means for us.** The shortcut we used has a known weakness in the closest field, and
the authors explain it as a basic limitation of the method rather than a tuning issue.

---

## 3. The standard fix: train the model to continue from known parts

Several papers train the model *directly* on "here is part of the sequence, fill in the rest",
by hiding random parts of the training examples:

* **Flexible Diffusion Models of Long Videos** (Harvey, Naderiparizi, Masrani, Weilbach and Wood,
  NeurIPS 2022). During training, randomly chosen frames are given and the model fills in others.
  One model can then continue, fill gaps, or plan a long video, and makes coherent videos far
  longer than its training clips.
* **MCVD** (Voleti, Jolicoeur-Martineau and Pal, NeurIPS 2022). During training, all past frames or
  all future frames are randomly hidden. One model then predicts forward, backward, in between, or
  from nothing, built from simple convolutional networks like ours.
* **CSDI** (Tashiro, Song, Song and Ermon, NeurIPS 2021). The same idea for time series (filling in
  missing sensor values): during training, known values are given and the model learns to fill in
  hidden ones. It is the closest to our kind of data.

**What it means for us.** This is our planned "retrain with known starts" step (E3). It is the
established solution, and it removes the need for the redo trick altogether.

---

## 4. Going very long without errors building up

* **Diffusion Forcing** (Chen, Martí Monsó, Du, Simchowitz, Tedrake and Sitzmann, NeurIPS 2024).
  Each step of the sequence gets its own noise level during training, so the model learns to work
  from slightly uncertain history. It can then keep generating well past its training length,
  where older methods drift off. (Our plan's "noisy history" setting borrows this idea.)
* **Rolling Diffusion Models** (Ruhe, Heek, Salimans and Hoogeboom, ICML 2024). A sliding window
  where later steps are noisier than earlier ones; as the first step is finished, a new one joins
  the window. This allows sequences of any length. Tested on video and fluid flow.

**What it means for us.** Ten years of rain is about 22,000 joins, far more than any video in
these papers. If the next run shows slow drift (rain changing steadily from year 1 to year 10),
these are the methods to follow, and both require training the model for it.

---

## 5. Weather and rain

* **GenCast** (Price and others, *Nature* 2024). Google DeepMind's weather model makes 15-day
  forecasts one 12-hour step at a time, each step **trained** to follow from the previous one.
  It beats the leading European forecasting system on most measures.
* **LDCast** (Leinonen, Hamann, Nerini, Germann and Franch, 2023). Rain nowcasting from radar: the
  model is **trained** to continue from the most recent radar images, and gets the forecast
  uncertainty right better than earlier methods.
* **Diffusion for zero-heavy rainfall** (for example the "Zero Inflation Diffusion Framework",
  2025): recent work treats the mostly-zero nature of rainfall as a problem needing special care
  in diffusion models, which supports our explanation of the dry bias.
* **Classical rainfall generators** (point-process and copula models, for example
  Neyman-Scott rectangular pulses): the established hydrology approach to synthetic sub-daily
  rainfall. Our classical baselines belong to this family.

I found no published diffusion model that generates **long continuous 5-minute rainfall records
for sewer or flood simulation**: existing diffusion work on rain is forecasting (minutes to days
ahead) or spatial downscaling. That gap is where this project sits.

---

## What this means for the thesis and the next step

1. **The next step is well supported.** Every successful long-sequence method in the literature
   *trains* the model to continue from known parts; none relies on the shortcut we tried.
2. **Our negative result fits a known pattern.** The shortcut (pasting the known part) is
   documented to give incoherent continuations, and the redo trick is proven to be biased.
3. **Possible original points** (to be stated cautiously, after a final check):
   * the redo trick's bias shows up as a strong **dry bias on mostly-zero data** such as rainfall;
     switching it off removes the bias while keeping storms going across joins (our 11-day test);
   * long continuous 5-minute rainfall from a diffusion model, evaluated for flood-control use.
4. **Order of work.** The quick test (head start without redoing) is still worth running, as it
   costs half a day and no code. But if it drifts over 10 years, the literature points clearly to
   retraining with known starts (Flexible Diffusion, MCVD, CSDI), with noisy history (Diffusion
   Forcing) if drift remains.

---

## References

* Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L. (2022). RePaint:
  Inpainting using Denoising Diffusion Probabilistic Models. *CVPR 2022*.
  [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html)
* Rout, L., Parulekar, A., Caramanis, C., Shakkottai, S. (2023). A Theoretical Justification for
  Image Inpainting using Denoising Diffusion Probabilistic Models. arXiv:2302.01217.
  [paper](https://arxiv.org/abs/2302.01217)
* Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D. J. (2022). Video Diffusion
  Models. *NeurIPS 2022*. [paper](https://arxiv.org/abs/2204.03458)
* Harvey, W., Naderiparizi, S., Masrani, V., Weilbach, C., Wood, F. (2022). Flexible Diffusion
  Modeling of Long Videos. *NeurIPS 2022*.
  [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/b2fe1ee8d936ac08dd26f2ff58986c8f-Paper-Conference.pdf)
* Voleti, V., Jolicoeur-Martineau, A., Pal, C. (2022). MCVD: Masked Conditional Video Diffusion
  for Prediction, Generation, and Interpolation. *NeurIPS 2022*.
  [paper](https://arxiv.org/abs/2205.09853)
* Tashiro, Y., Song, J., Song, Y., Ermon, S. (2021). CSDI: Conditional Score-based Diffusion Models
  for Probabilistic Time Series Imputation. *NeurIPS 2021*.
  [paper](https://www.semanticscholar.org/paper/CSDI:-Conditional-Score-based-Diffusion-Models-for-Tashiro-Song/8982bb695dcebdacbfd079c62cd7acca8a8b48dc)
* Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V. (2024). Diffusion
  Forcing: Next-token Prediction Meets Full-Sequence Diffusion. *NeurIPS 2024*.
  [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2aee1c4159e48407d68fe16ae8e6e49e-Paper-Conference.pdf)
* Ruhe, D., Heek, J., Salimans, T., Hoogeboom, E. (2024). Rolling Diffusion Models. *ICML 2024*.
  [paper](https://proceedings.mlr.press/v235/ruhe24a.html)
* Price, I., Sanchez-Gonzalez, A., Alet, F., et al. (2024). Probabilistic weather forecasting with
  machine learning. *Nature* 637, 84–90. [preprint](https://arxiv.org/abs/2312.15796)
* Leinonen, J., Hamann, U., Nerini, D., Germann, U., Franch, G. (2023). Latent diffusion models
  for generative precipitation nowcasting with accurate uncertainty quantification.
  arXiv:2304.12891. [paper](https://arxiv.org/abs/2304.12891)
* From Noise to Precision: A Diffusion-Driven Approach to Zero-Inflated Precipitation Prediction
  (2025). arXiv:2509.10501. [paper](https://arxiv.org/abs/2509.10501)
