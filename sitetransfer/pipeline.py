"""The pipeline commands. Each reads the site description, writes its outputs and one run record.

Order for a transfer site:  build-data -> derive -> labels -> train-config -> baseline -> pilot -> bands
-> evaluate; the home site needs only derive -> pilot -> bands -> evaluate; verdict reads both.
"""
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, checks, labels, metrics as M, pilot, rules, training
from . import ingest_singapore as ing
from . import timegrid as tg
from .provenance import ROOT, RunRecord, rel, sha256_file, write_json
from .site import Site, SiteError, clean_series, load_generated

GEN_YEARS = 10
GEN_SEED = 42


def _record(site, command, params):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RunRecord(command, site.name, params, site.results_dir / "records" / f"{stamp}_{command}.json")


def _require_transfer(site, command):
    if site.is_home:
        raise SiteError(f"`{command}` is for a transfer site; {site.name} is the home site")


# ── build-data ───────────────────────────────────────────────────────────────────────────────────

def build_data(site):
    _require_transfer(site, "build-data")
    cfg = site.cfg["ingest"]
    y0, y1 = cfg["years"]
    years = list(range(y0, y1 + 1))
    first_year = site.first_year
    n = len(years) * tg.STEPS_PER_YEAR
    out_csv = site.path(site.cfg["series"]["csv"])
    out_dir = out_csv.parent
    with _record(site, "build-data", {"ingest": cfg}) as rec:
        raw = []
        for y in years:
            path = ROOT / cfg["raw_pattern"].format(year=y)
            rec.input(path)
            raw.append(ing.read_raw_year(path, y, first_year, n))
            print(f"read {y}: {raw[-1].counts}", flush=True)
        series, meta = ing.assemble(raw, n)
        stats = ing.station_year_stats(series, years, first_year, cfg["gauge_resolution_mm"])
        u = cfg["usable_stations"]
        usable = ing.usable_stations(stats, years, u["min_completeness"], u["exempt_years"], u["min_resolution_share"])
        mean_comp = stats.groupby("station_id")["completeness"].mean().to_dict()
        dist = ing.km_matrix(meta.loc[usable, "lon"], meta.loc[usable, "lat"])
        cluster, spread = ing.compact_cluster(usable, dist, mean_comp, cfg["cluster_size"])
        values, flags, G, fill_counts = ing.build_cluster_series(series, meta, cluster, usable, cfg)

        grid = cfg["gauge_resolution_mm"] / cfg["cluster_size"]
        assert np.all(values >= 0), "negative rain"
        assert np.allclose(values / grid, np.round(values / grid), atol=1e-6), "value off the gauge grid"
        assert np.all(values[flags == 3] == 0), "outage placeholder must be zero"

        dates = tg.slot_starts(n, first_year).strftime("%Y-%m-%d %H:%M:%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"date": dates, "avg_rainfall": values,
                           "fill_flag": pd.Series(flags).map(ing.FLAG_NAMES).to_numpy(),
                           "outage": (flags == 3).astype(np.int8)})
        df.to_csv(out_csv, index=False)
        gauges_npz = out_dir / "cluster_gauges.npz"
        np.savez_compressed(gauges_npz, station_ids=np.array(cluster),
                            observed=np.stack([series[s] for s in cluster], axis=1), filled=G.astype(np.float32))

        obs = flags != 3
        hr = (np.arange(n) % tg.STEPS_PER_DAY) // tg.STEPS_PER_HOUR
        hourly_mean = np.bincount(hr[obs], weights=values[obs], minlength=24) / np.bincount(hr[obs], minlength=24)
        audit = {
            "read": {y: r.counts for y, r in zip(years, raw)},
            "stations_seen": int(len(series)),
            "usable_rule": u, "usable_stations": usable,
            "cluster": [{"station_id": s, "name": meta.loc[s, "name"], "lon": float(meta.loc[s, "lon"]),
                         "lat": float(meta.loc[s, "lat"]), "mean_completeness": round(mean_comp[s], 4)} for s in cluster],
            "cluster_max_spacing_km": round(spread, 3),
            "fill_counts_per_gauge": fill_counts,
            "flag_share_by_year": {y: {ing.FLAG_NAMES[f]: round(float((flags[tg.year_slice(k)] == f).mean()), 5)
                                       for f in ing.FLAG_NAMES} for k, y in enumerate(years)},
            "outage_runs_by_year": {y: {"count": int(len(r)), "longest_hours": round(float(r.max()) / 12, 2) if len(r) else 0}
                                    for k, y in enumerate(years)
                                    for r in [ing.outage_runs(flags[tg.year_slice(k)])]},
            "diurnal_peak_hour_local": int(np.argmax(hourly_mean)),
            "diurnal_peak_over_trough": round(float(hourly_mean.max() / max(hourly_mean.min(), 1e-12)), 2),
            "value_grid_mm": grid,
            "station_year_stats": stats.round(4).to_dict(orient="records"),
        }
        write_json(out_dir / "audit.json", audit)
        for p in (out_csv, gauges_npz, out_dir / "audit.json"):
            rec.output(p)
        rec.result(cluster=cluster, spacing_km=spread, n_usable=len(usable),
                   flag_share={ing.FLAG_NAMES[f]: round(float((flags == f).mean()), 5) for f in ing.FLAG_NAMES},
                   diurnal_peak_hour=audit["diurnal_peak_hour_local"])
    return audit


# ── derive ───────────────────────────────────────────────────────────────────────────────────────

def _home_depth_share(home):
    x, _ = home.select_years(home.years("train"))
    _, dep = rules.storms_2h(x)
    return rules.depth_share(dep, home.cfg["frozen"]["storm_depth_mm"])


def derive(site):
    """Apply every rule to the site's training years and write derived.json (home site: check invariant 1)."""
    home = Site.load("astlingen")
    with _record(site, "derive", {}) as rec:
        x, mask, years = site.load_record(clean=False)
        train = site.years("train")
        sel = np.concatenate([np.arange(*tg.year_slice(years.index(y)).indices(len(x))) for y in train])
        xt, mt = x[sel], mask[sel]
        obs = xt[~mt]
        tau = rules.wet_threshold(obs)
        s = rules.asinh_scale(obs, tau)
        share = _home_depth_share(home)
        outage = rules.outage_share_by_year(mask, site.first_year)
        if site.is_home:
            ref_years = list(site.cfg["frozen"]["reference_years"])
        else:
            ref_years = rules.reference_years(outage, train, site.cfg["reference_rule"]["max_outage_share"])
        ref_sel = np.concatenate([np.arange(*tg.year_slice(years.index(y)).indices(len(x))) for y in ref_years])
        _, dep = rules.storms_2h(clean_series(np.where(mask[ref_sel], 0.0, x[ref_sel]), tau))
        depth = rules.storm_depth_threshold(dep, share)
        derived = {"params": {"wet_threshold_mm": tau, "asinh_scale_mm": s, "storm_depth_mm": depth,
                              "reference_years": ref_years},
                   "facts": {"resolution_mm": rules.resolution(obs), "home_deep_storm_share": share,
                             "outage_share_by_year": outage, "training_years": train,
                             "storms_per_year_reference": len(dep) / len(ref_years)},
                   "rules": "sitetransfer/rules.py (wet_threshold, asinh_scale, storm_depth_threshold, reference_years)"}
        if site.is_home:
            frozen = site.cfg["frozen"]
            x_ref, _ = site.select_years(ref_years)
            _, dep_home = rules.storms_2h(x_ref)
            checks_ = {
                "wet_threshold_mm": tau == frozen["wet_threshold_mm"],
                "asinh_scale_mm": round(s, 6) == frozen["asinh_scale_mm"],
                "storm_depth_classification": int((dep_home >= depth).sum()) == int((dep_home >= frozen["storm_depth_mm"]).sum()),
            }
            derived["invariant_1"] = checks_
            if not all(checks_.values()):
                raise AssertionError(f"invariant 1 failed on the home site: {checks_}")
        out = write_json(site.derived_path, derived)
        rec.output(out)
        rec.result(**derived["params"])
    return derived


# ── labels and splits ────────────────────────────────────────────────────────────────────────────

EXPORT_COLS = ["datetime", "date", "avg_rainfall", "rainfall_intensity", "hmm_state", "day_of_year",
               "annual_interval_idx", "fill_flag", "outage"]


def make_labels(site):
    _require_transfer(site, "labels")
    with _record(site, "labels", {"n_states": labels.N_STATES, "smooth_window": labels.SMOOTH_WINDOW}) as rec:
        src = site.path(site.cfg["series"]["csv"])
        rec.input(src)
        df = pd.read_csv(src)
        x, mask, years = site.load_record(clean=False)
        train = site.years("train")
        sel = np.concatenate([np.arange(*tg.year_slice(years.index(y)).indices(len(x))) for y in train])
        ph = labels.fit_phases(labels.slot_climatology(x[sel], mask[sel]))
        states = labels.label_steps(ph["profile"], len(x))
        idx = np.arange(len(x)) % tg.STEPS_PER_YEAR
        df["datetime"] = df["date"]
        df["rainfall_intensity"] = df["avg_rainfall"]
        df["hmm_state"] = states
        df["annual_interval_idx"] = idx
        df["day_of_year"] = idx // tg.STEPS_PER_DAY + 1
        split_dir = site.path("splits_dir")
        split_dir.mkdir(parents=True, exist_ok=True)
        year_of = np.arange(len(x)) // tg.STEPS_PER_YEAR + site.first_year
        manifest = {"source": rel(src), "source_sha256": sha256_file(src), "phases": {
            "min_days": ph["min_days"], "transitions_per_year": ph["n_transitions"], "state_means": ph["state_means"],
            "phase_start_days": _phase_starts(ph["profile"])}, "splits": {}}
        for split, fname in (("train", "train_years_labelled.csv"), ("val", "val_years_labelled.csv"),
                             ("test", "test_years_labelled.csv")):
            ys = site.years(split)
            part = df.loc[np.isin(year_of, ys), EXPORT_COLS]
            path = split_dir / fname
            part.to_csv(path, index=False)
            rec.output(path)
            manifest["splits"][split] = {"file": fname, "years": ys, "rows": int(len(part)), "sha256": sha256_file(path),
                                         "outage_steps": int(part["outage"].sum()),
                                         "phase_distribution": part["hmm_state"].value_counts().sort_index().to_dict()}
        bundle = labels.transition_bundle(states[sel], ph["profile"], ph["min_days"], block_size=64)
        bpath = site.path("transition_bundle")
        with open(bpath, "wb") as f:
            pickle.dump(bundle, f)
        rec.output(bpath)
        write_json(split_dir / "MANIFEST.json", manifest)
        rec.output(split_dir / "MANIFEST.json")
        rec.result(**manifest["phases"])
    return manifest


def _phase_starts(profile):
    p = np.asarray(profile)
    ch = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
    return [{"day": int(i // tg.STEPS_PER_DAY) + 1, "phase": int(p[i])} for i in ch]


# ── training config ──────────────────────────────────────────────────────────────────────────────

def train_config(site, run_id):
    _require_transfer(site, "train-config")
    home = Site.load("astlingen")
    with _record(site, "train-config", {"run_id": run_id}) as rec:
        cfg, changes, facts = training.build_config(site, home, run_id)
        out = ROOT / "regime_training" / f"config_{run_id}.yaml"
        out.write_text(training.render(cfg, changes, facts, site, run_id))
        import yaml
        written = yaml.safe_load(out.read_text())
        base = yaml.safe_load(training.TEMPLATE.read_text())
        extra = set(training.diff_keys(base, written)) - training.ALLOWED_CHANGES
        if extra:
            raise AssertionError(f"invariant 9: config differs from v14 outside the allowed keys: {extra}")
        rec.input(training.TEMPLATE)
        rec.output(out)
        rec.result(changes=changes, **facts)
    return out, changes, facts


# ── baseline ─────────────────────────────────────────────────────────────────────────────────────

def baseline(site, p=64):
    _require_transfer(site, "baseline")
    with _record(site, "baseline", {"model": "occurrence-matched latent Gaussian AR", "p": p,
                                    "years": GEN_YEARS, "seed": GEN_SEED}) as rec:
        x, mask = site.select_years(site.years("train"))
        model = baselines.fit(x, site.tau, p=p, mask=mask)
        sim = baselines.generate(model, GEN_YEARS, GEN_SEED)
        out = site.series_to_score()["lgar"]
        out.parent.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range("2026-01-01 00:00:00", periods=len(sim), freq="5min")
        pd.DataFrame({"date": dates, "avg_rainfall": sim}).to_csv(out, index=False)
        card = {"model": "occurrence-matched latent Gaussian AR", "p": p, "wet_threshold_mm": site.tau,
                "trained_on_years": site.years("train"), "outage_steps_excluded": int(mask.sum()),
                "latent_rho": {k: float(model["rho"][k - 1]) for k in (1, 2, 3, 6, 12, 24, 36, 64) if k <= p},
                "toeplitz_min_eigenvalue": model["min_eig"], "innovation_sigma": model["sigma"],
                "sum_phi": float(model["phi"].sum()), "zero_threshold_z": float(model["eq"].z_threshold),
                "generated_zero_pct": float(100 * (sim == 0).mean()),
                "generated_mm_per_year": float(sim.sum() / GEN_YEARS)}
        cpath = write_json(site.results_dir / "lgar_model_card.json", card)
        rec.output(out)
        rec.output(cpath)
        rec.result(**{k: card[k] for k in ("generated_zero_pct", "generated_mm_per_year", "toeplitz_min_eigenvalue")})
    return card


# ── pilot ────────────────────────────────────────────────────────────────────────────────────────

KEY_METRICS = ["volume_mm_yr", "long_storms_yr", "long_storm_rain_pct", "storms_ge_depth_yr",
               "rain_in_320min_pct", "top_storm_mm", "var_scaling_rmse", "w1_daily", "diurnal_r",
               "diurnal_peak_trough", "monthly_r"]


def run_pilot(site, seeds=(0, 1, 2)):
    with _record(site, "pilot", {"variants": pilot.VARIANTS, "seeds": list(seeds), "years": GEN_YEARS}) as rec:
        x, mask = site.reference()
        P = M.Params(site.params["storm_depth_mm"])
        R = M.reference(x, P, mask=mask if mask.any() else None)
        out_dir = site.results_dir / "pilot"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        for name, kw in pilot.VARIANTS.items():
            per_seed = []
            for seed in seeds:
                y, info = pilot.perfect_window(x, mask, n_years=GEN_YEARS, seed=seed, **kw)
                path = out_dir / f"{name}_s{seed}.npy"
                np.save(path, y.astype(np.float32))
                rec.output(path)
                h = M.hydro(y, R, P)
                per_seed.append({k: h[k] for k in KEY_METRICS} | info)
            summary[name] = {"per_seed": per_seed,
                             "min": {k: float(min(s[k] for s in per_seed)) for k in KEY_METRICS},
                             "max": {k: float(max(s[k] for s in per_seed)) for k in KEY_METRICS}}
            print(name, {k: round(summary[name]["min"][k], 3) for k in ("long_storms_yr", "rain_in_320min_pct",
                                                                         "diurnal_r")}, flush=True)
        real = M.hydro(x, R, P, mask=mask if mask.any() else None)
        summary["real_reference"] = {k: real[k] for k in KEY_METRICS}
        out = write_json(site.results_dir / "pilot.json", summary)
        rec.output(out)
    return summary


# ── bands ────────────────────────────────────────────────────────────────────────────────────────

def bands(site, n_draws=1000, workers=1, seed=0):
    with _record(site, "bands", {"n_draws": n_draws, "n_years": GEN_YEARS, "seed": seed}) as rec:
        x, mask = site.reference()
        P = M.Params(site.params["storm_depth_mm"])
        b = checks.resampled_bands(x, mask, P, n_draws=n_draws, n_years=GEN_YEARS, seed=seed, workers=workers)
        out = write_json(site.results_dir / "bands.json", {"n_draws": n_draws, "n_years": GEN_YEARS, "seed": seed,
                                                           "reference_years": site.params["reference_years"],
                                                           "bands": b})
        rec.output(out)
    return b


def load_bands(site):
    path = site.results_dir / "bands.json"
    if not path.exists():
        raise SiteError(f"{path} missing: run `python -m sitetransfer bands --site {site.name}`")
    raw = json.loads(path.read_text())["bands"]
    return {k: (v if isinstance(v, dict) else tuple(v)) for k, v in raw.items()}


# ── evaluate ─────────────────────────────────────────────────────────────────────────────────────

def evaluate(site):
    with _record(site, "evaluate", {}) as rec:
        x, mask = site.reference()
        P = M.Params(site.params["storm_depth_mm"])
        R = M.reference(x, P, mask=mask if mask.any() else None)
        rb = load_bands(site)
        rec.input(site.results_dir / "bands.json")
        series = {}
        hx, hm = site.heldout()
        series["real_heldout"] = (hx, hm if hm.any() else None)
        for name, path in site.series_to_score().items():
            if path.exists():
                rec.input(path)
                series[name] = (load_generated(path, site.tau), None)
            else:
                print(f"skip {name}: {rel(path)} not found", flush=True)
        for p in sorted((site.results_dir / "pilot").glob("*_s0.npy")):
            series[f"pilot_{p.stem[:-3]}"] = (clean_series(np.load(p), site.tau), None)
        result = {"site": site.name, "params": site.params, "series": {}}
        for name, (a, m) in series.items():
            h = M.hydro(a, R, P, mask=m)
            fixed = checks.score(h, R, "fixed")
            resampled = checks.score(h, R, rb)
            result["series"][name] = {
                "metrics": h,
                "fixed": {"checks": fixed, "gate_a": sum(bool(fixed[c]["pass"]) for c in checks.GATE_A_18),
                          "tier6": sum(bool(fixed[c]["pass"]) for c in checks.TIER_6)},
                "resampled": {"checks": resampled, "families": checks.families(resampled)},
            }
            fam = result["series"][name]["resampled"]["families"]
            print(f"{name:22s} Gate A (fixed) {result['series'][name]['fixed']['gate_a']:2d}/18 | families: "
                  + " ".join(f"{f}:{'P' if v['pass'] else ('F' if v['pass'] is not None else '-')}"
                             f"({v['passed']}/{v['of']})" for f, v in fam.items()), flush=True)
        out = write_json(site.results_dir / "evaluation.json", result)
        rec.output(out)
    return result


# ── verdict ──────────────────────────────────────────────────────────────────────────────────────

def verdict(home_name="astlingen", site_name="singapore"):
    home, site = Site.load(home_name), Site.load(site_name)
    ev = {}
    for s in (home, site):
        path = s.results_dir / "evaluation.json"
        if not path.exists():
            raise SiteError(f"{path} missing: run `python -m sitetransfer evaluate --site {s.name}`")
        ev[s.name] = json.loads(path.read_text())
    H, S = ev[home.name]["series"], ev[site.name]["series"]
    lines = [f"# Transfer verdict: {home.name} -> {site.name}", "",
             f"Written {dt.datetime.now().isoformat(timespec='seconds')} by `python -m sitetransfer verdict`. "
             "Rules: code_plan/SINGAPORE_TRANSFER_PLAN.md, sections 8.4 and 8.5. Bands: resampled at both sites.", ""]
    out = {"home": home.name, "site": site.name}
    if "v14" not in S:
        lines.append(f"**Recipe verdict: not decidable yet** (no generated v14 series at {site.name}).")
        out["recipe"] = None
    else:
        hf, sf = H["v14"]["resampled"]["families"], S["v14"]["resampled"]["families"]
        vol = S["v14"]["resampled"]["checks"]["T1 volume"]["value"]
        status, why, lost = checks.recipe_verdict(hf, sf, vol)
        out["recipe"] = {"status": status, "why": why, "lost": lost}
        lines += [f"**Recipe verdict: {status}.** {why}.", "", "| Family | " + f"v14 at {home.name} | v14 at {site.name} |",
                  "| :--- | :---: | :---: |"]
        for f in checks.FAMILIES:
            tag = " (predicted, not required)" if f in checks.PREDICTED_FAMILIES else ""
            lines.append(f"| {f}{tag} | {_fam(hf[f])} | {_fam(sf[f])} |")
    lines += ["", "## Relative standing against the latent Gaussian AR (families B and C)", ""]
    for name, E in ((home.name, H), (site.name, S)):
        if "v14" in E and "lgar" in E:
            row = {f: (E["v14"]["resampled"]["families"][f]["passed"], E["lgar"]["resampled"]["families"][f]["passed"])
                   for f in ("B", "C")}
            lines.append(f"- {name}: B v14 {row['B'][0]} vs AR {row['B'][1]}; C v14 {row['C'][0]} vs AR {row['C'][1]}")
    lines += ["", "## Window horizon (family D against the perfect 5h20 generator)", ""]
    for name, s_obj, E in ((home.name, home, H), (site.name, site, S)):
        pj = s_obj.results_dir / "pilot.json"
        if pj.exists() and "v14" in E:
            pl = json.loads(pj.read_text())
            for k in ("long_storms_yr", "rain_in_320min_pct", "var_scaling_rmse"):
                lo, hi = pl["w64"]["min"][k], pl["w64"]["max"][k]
                v = E["v14"]["metrics"][k]
                lines.append(f"- {name} {k}: v14 {v:.3g}; perfect 5h20 window {lo:.3g}-{hi:.3g}; "
                             f"real {pl['real_reference'][k]:.3g}")
    path = ROOT / "results" / "transfer" / "verdict.md"
    path.write_text("\n".join(lines) + "\n")
    write_json(path.with_suffix(".json"), out)
    return path


def _fam(f):
    if f["pass"] is None:
        return "no band"
    return f"{'pass' if f['pass'] else 'fail'} ({f['passed']}/{f['of']})"
