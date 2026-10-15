"""Invariants on built data and configs (`python -m tworesolution check`); with --torch, also that the fine model
starts as v14 exactly with the real v14 weights. The code-level invariants are the unit tests in tests/.
"""
import numpy as np
import pandas as pd
import yaml

from sitetransfer.training import diff_keys

from . import data as D
from .grid import FACTOR, STEPS_PER_YEAR_5, STEPS_PER_YEAR_30, aggregate, two_level


def _ok(name, cond, detail=""):
    return {"invariant": name, "ok": bool(cond), "detail": detail}


def check_data():
    out = []
    profile5 = D.calendar5()
    for name in D.SPLIT_YEARS:
        x, states = D.load_split(name)
        out.append(_ok(f"{name}: seasonal phases are the calendar profile",
                       np.array_equal(states, np.tile(profile5, len(x) // STEPS_PER_YEAR_5))))
    now, saved = D.derive(), D.load_derived()
    out.append(_ok("derived.json matches the data", all(np.isclose(now[k], saved[k]) for k in now),
                   f"tau30 {now['tau30_mm']} mm, asinh scale {now['asinh_scale30_mm']} mm"))
    for name in D.SPLIT_YEARS:
        x, _ = D.load_split(name)
        x5c, x30 = two_level(x, saved["tau5_mm"], saved["tau30_mm"])
        df = pd.read_csv(D.SPLITS30 / f"{name}_years_labelled.csv")
        out += [_ok(f"{name}: 30-minute file = half-hour totals of the split", np.allclose(df["avg_rainfall"], x30,
                                                                                          rtol=0, atol=1e-9)),
                _ok(f"{name}: 30-minute phases = calendar every 6th step", np.array_equal(
                    df["hmm_state"], np.tile(profile5[::FACTOR], len(x30) // STEPS_PER_YEAR_30))),
                _ok(f"{name}: six 5-minute values of each half-hour sum to its total",
                    np.allclose(aggregate(x5c), x30, rtol=0, atol=1e-12)),
                _ok(f"{name}: wet totals >= tau30", (x30[x30 > 0] >= saved["tau30_mm"]).all())]
    c = D.config_dicts(saved)
    for path, want in ((D.COARSE_CONFIG, c["coarse"]), (D.COARSE_LONG_CONFIG, c["coarse_long"]),
                       (D.FINE_CONFIG, c["fine"])):
        have = yaml.safe_load(path.read_text()) if path.exists() else {}
        out.append(_ok(f"{path.name} is v14's config with only the allowed keys changed", not diff_keys(have, want),
                       f"differs in {diff_keys(have, want)}" if diff_keys(have, want) else ""))
    return out


def check_v14_start(device="cpu", n=6):
    """The fine model built for training (v14's EMA weights + zero condition layer) samples exactly as v14."""
    import torch

    from .network import fine_from_v14, load_args, load_plain
    from .sampling import heun, schedule
    v14 = load_plain(load_args(D.V14_CONFIG, device), D.V14_RUN / "best_regime_model.pt", device)
    fine = fine_from_v14(load_args(D.FINE_CONFIG, device), D.V14_RUN / "best_regime_model.pt", device).eval()
    g = torch.Generator().manual_seed(0)
    xT = torch.randn(n, 1, 8, 8, generator=g).to(device)
    cond = torch.randn(n, 1, 8, 8, generator=g).to(device)
    lab = torch.eye(4, device=device)[torch.arange(n, device=device) % 4]
    t = schedule(v14.net, v14.args.diffusion_steps, device)
    with v14.ema_scope():
        a = heun(v14.net, xT, lab, t)
    with fine.ema_scope():
        b = heun(fine.net, xT, lab, t, cond=cond)
    return [_ok("fine model at the start samples exactly as v14 (any condition)", torch.equal(a, b),
                f"max difference {float((a - b).abs().max()):.3g}")]


def report(results):
    bad = [r for r in results if not r["ok"]]
    for r in results:
        print(f"  {'ok  ' if r['ok'] else 'FAIL'}  {r['invariant']}" + (f"  ({r['detail']})" if r["detail"] else ""))
    return not bad
