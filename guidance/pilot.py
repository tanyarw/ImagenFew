"""The guidance pilot (code_plan/GUIDANCE_PLAN.md, section 4): settings, chained generation, scoring, the rules.

Every setting generates the same 256 chains: 8 start dates spread over the calendar year, 32 chains each, 20
days per chain (with 32 chains the run-to-run noise of total rain was ~16%, wider than the 10% rule), calendar seasonal phases, plan seed 42. All settings consume the random stream identically
(guidance adds gradients, not random draws), so settings with the same context length are paired.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from sitetransfer.provenance import ROOT, write_json

from .sampler import check_weights, sample_piece

STEPS_PER_DAY = 288
STEPS_PER_YEAR = 105_120
L = 64                                   # one piece, 5 h 20 min
WET_MM = 0.005                           # Astlingen wet threshold
WET_WEIGHT = 10.0                        # ~ dry : wet share of steps (91 : 9)
GRID = [0.003, 0.006, 0.0125, 0.025, 0.05, 0.1]   # strengths; the tiny smoke run (2026-10-14) overshot from 0.025 up
SEED = 42
OUT = Path(os.environ.get("GUIDANCE_OUT", ROOT / "results" / "guidance"))   # override only for smoke tests
CALENDAR = ROOT / "data" / "rainfall" / "splits" / "seasonal_transition_matrix_train_len64.pkl"

# index -> (name, context columns, variant, strength). Index 0 is the amount reference (independent pieces).
TASKS = [("independent", 0, "plain", 0.0), ("paste", 2, "plain", 0.0)]
TASKS += [(f"{v}_w{w}", 2, v, w) for v in ("plain", "wet", "rainy") for w in GRID]

RULES = {"continuation_min": 0.70, "wet_share_pp": 1.0, "volume_ratio": (0.90, 1.10), "p999_ratio": (0.85, 1.15)}
TWO_HOUR_RULE = "2 h adopted if it passes all four rules and its continuation beats the 80-min setting by >= 0.03"


# ── model ────────────────────────────────────────────────────────────────────────────────────────

def load_v14(device):
    import regime_training.generate_context as gc
    from regime_training.generate_hmm_v1 import DiffusionProcess
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "regime_training" / "config_v14.yaml"), resolve=True)
    args = argparse.Namespace(**cfg)
    args.device = device
    model, scaler = gc.load_model(argparse.Namespace(run="v14"), args)
    net = model.net
    proc = DiffusionProcess(args, net, (1, args.img_resolution, args.img_resolution))
    smin, smax = max(proc.sigma_min, net.sigma_min), min(proc.sigma_max, net.sigma_max)
    idx = torch.arange(proc.num_steps, dtype=torch.float64, device=device)
    t = (smax ** (1 / proc.rho) + idx / (proc.num_steps - 1) * (smin ** (1 / proc.rho) - smax ** (1 / proc.rho))) ** proc.rho
    t_steps = torch.cat([net.round_sigma(t), torch.zeros_like(t[:1])])
    return model, scaler, net, t_steps


def pilot_offsets(starts=8, chains_per_start=32):
    days = [int(round(k * 365 / starts)) for k in range(starts)]
    return np.repeat(np.array(days) * STEPS_PER_DAY, chains_per_start)


# ── generation ───────────────────────────────────────────────────────────────────────────────────

def generate(model, scaler, net, t_steps, offsets, n_steps, context_cols, variant, w, seed=SEED, device="cpu"):
    """Chained generation for chains starting at calendar positions `offsets`. Returns (mm [M, n], seams)."""
    with open(CALENDAR, "rb") as f:
        profile = np.asarray(pickle.load(f)["climatology_profile_1yr"])
    M, K = len(offsets), context_cols
    known_steps = 8 * K
    z_wet = float(scaler.transform(np.array([[WET_MM]]))[0, 0])
    eye = torch.eye(4, device=device)
    mask_ts = torch.zeros(M, L, 1, device=device)
    mask_ts[:, :known_steps] = 1
    mask = model.ts_to_img(mask_ts).to(torch.float64).to(device)
    out = np.zeros((M, n_steps))
    seams, rel_trace = [], []
    torch.manual_seed(seed)
    pos, ctx = 0, None
    with model.ema_scope():
        while pos < n_steps:
            known = torch.zeros(M, L, 1, dtype=torch.float64, device=device)
            if ctx is None or K == 0:
                m, first_new, wi = torch.zeros_like(mask), 0, 0.0
            else:
                known[:, :known_steps, 0] = ctx
                m, first_new, wi = mask, known_steps, w
                seams.append(pos)
            x_known = model.ts_to_img(known).to(torch.float64).to(device)
            n_new = L - first_new
            t_mid = min(pos + n_new // 2, n_steps - 1)
            labels = eye[torch.as_tensor(profile[(offsets + t_mid) % STEPS_PER_YEAR], device=device)]
            weight, active = check_weights(x_known, m, z_wet, variant, WET_WEIGHT)
            trace = [] if wi else None
            img = sample_piece(net, x_known, m, labels, t_steps, wi, weight, active, trace)
            if trace:
                rel_trace.append([r for _, r in trace])
            ts = model.img_to_ts(img.to(torch.float32))[:, :, 0].to(device, torch.float64)
            take = min(n_new, n_steps - pos)
            new = ts[:, first_new:first_new + take].cpu().numpy()
            out[:, pos:pos + take] = scaler.inverse_transform(new.reshape(-1, 1)).reshape(M, take)
            pos += take
            ctx = ts[:, L - known_steps:] if K else None
    out[out < WET_MM] = 0.0
    trace = np.mean(rel_trace, axis=0).tolist() if rel_trace else []
    return out, np.array(seams, dtype=int), trace


# ── scoring ──────────────────────────────────────────────────────────────────────────────────────

def score(out, seams):
    wet = out > 0
    M, n = out.shape
    s = seams[seams > 0]
    prev, here = wet[:, s - 1], wet[:, s]
    inside = np.setdiff1d(np.arange(1, n), s)
    surv = {}
    for h, k in (("1h", 12), ("2h", 24), ("4h", 48)):
        ok = [bool(wet[i, p:p + k].all()) for i in range(M) for p in s if wet[i, p - 1] and p + k <= n]
        surv[h] = float(np.mean(ok)) if ok else float("nan")
    pos = out[out > 0]
    per_chain_mm = out.sum(axis=1) / (n / STEPS_PER_YEAR)
    per_chain_wet = 100 * wet.mean(axis=1)
    return {"continuation": float(here[prev].mean()) if prev.any() else float("nan"),
            "chains": int(M),
            "mm_per_year_se_pct": float(100 * per_chain_mm.std(ddof=1) / np.sqrt(M) / max(per_chain_mm.mean(), 1e-12)),
            "wet_share_se_pp": float(per_chain_wet.std(ddof=1) / np.sqrt(M)),
            "wet_seams": int(prev.sum()),
            "continuation_inside": float(wet[:, inside][wet[:, inside - 1]].mean()),
            "wet_share_pct": float(100 * wet.mean()),
            "mm_per_year": float(out.sum() / (M * n / STEPS_PER_YEAR)),
            "p999_mm": float(np.percentile(pos, 99.9)) if len(pos) else 0.0,
            "storm_survival_after_wet_seam": surv}


def judge(m, ref):
    """The four pass rules against the independent-piece reference."""
    checks = {"continuation >= 0.70": m["continuation"] >= RULES["continuation_min"],
              "wet share within 1 pp": abs(m["wet_share_pct"] - ref["wet_share_pct"]) <= RULES["wet_share_pp"],
              "volume within 10%": RULES["volume_ratio"][0] <= m["mm_per_year"] / ref["mm_per_year"] <= RULES["volume_ratio"][1],
              "P99.9 within 15%": RULES["p999_ratio"][0] <= m["p999_mm"] / ref["p999_mm"] <= RULES["p999_ratio"][1]}
    return {k: bool(v) for k, v in checks.items()}


def choose(results):
    """The passing setting that carries rain across seams best (highest continuation); ties: the smaller
    strength, then rainy before wet before plain (fewest interventions). Rule changed on 2026-10-14, before any
    pilot ran, from "smallest passing strength", which could pick a setting that barely clears 0.70."""
    order = {"rainy": 0, "wet": 1, "plain": 2}
    passing = [(-round(r["continuation"], 3), r["w"], order[r["variant"]], name) for name, r in results.items()
               if r["w"] > 0 and r.get("pass") and all(r["pass"].values())]
    return min(passing)[3] if passing else None


def task_paths(name):
    return OUT / "pilot" / f"{name}.npz", OUT / "pilot" / f"{name}.json"
