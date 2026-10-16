"""regime_training/config_v19.yaml: config_v14.yaml with only the window keys (and, by the plan's rule, the
epochs and batch size) changed.

The window keys match the user's 24x24 pretrained checkpoint: 288 steps (one day at 5 minutes), delay = embedding
= 24 (an image of 24 rows x 12 columns, padded to 24 x 24; the padding is marked known and never scored), attention
at 12, 6 and 3. Everything else (asinh transform and scale, seasonal phases, loss, optimiser, EMA, dropout, data
split, sampler) is v14's.
"""
import yaml

from sitetransfer.provenance import ROOT, rel, sha256_file
from sitetransfer.training import diff_keys

TEMPLATE = ROOT / "regime_training" / "config_v14.yaml"
CONFIG = ROOT / "regime_training" / "config_v19.yaml"
RUN_ID = "v19"
RUN_DIR = ROOT / "logs" / "ImagenFew" / "Rainfall_Regime" / RUN_ID
CHECKPOINT = ROOT / "models_ckpt" / "ImagenFew" / "ImagenFew_288_1200epochs.ckpt"
CHECKPOINT_SHA256 = "8268872cb24da3debe0ece55c9b3741e670be227367b0a52c107bfc7177505fd"
CALENDAR = ROOT / "data" / "rainfall" / "splits" / "seasonal_transition_matrix_train_len288.pkl"
WINDOW = {"seq_len": 288, "delay": 24, "embedding": 24, "img_resolution": 24, "attn_resolution": [12, 6, 3]}
ALLOWED = set(WINDOW) | {"run_dir", "epochs", "batch_size"}


def build(epochs=500, batch_size=2048):
    base = yaml.safe_load(TEMPLATE.read_text())
    cfg = dict(base) | WINDOW | {"epochs": int(epochs), "batch_size": int(batch_size),
                                 "run_dir": f"ImagenFew/Rainfall_{RUN_ID}/"}
    extra = set(diff_keys(base, cfg)) - ALLOWED
    if extra:
        raise ValueError(f"v19 config changes keys outside the allowed list: {sorted(extra)}")
    return base, cfg


def write(epochs=500, batch_size=2048, reason="v14's recipe"):
    base, cfg = build(epochs, batch_size)
    head = ["# " + "=" * 98,
            f"# v19: the v14 recipe with a one-day window. WRITTEN BY `python -m oneday config`; do not edit.",
            f"# Template: {rel(TEMPLATE)} (sha256 {sha256_file(TEMPLATE)[:16]}...). Only these keys differ:"]
    head += [f"#   {k}: {cfg[k]}" for k in diff_keys(base, cfg)]
    head += [f"# Epochs and batch size: {reason}.",
             f"# Initial weights: {rel(CHECKPOINT)} (sha256 {CHECKPOINT_SHA256[:16]}...).",
             "# Plan: code_plan/ONE_DAY_WINDOW_PLAN.md.", "# " + "=" * 98, ""]
    CONFIG.write_text("\n".join(head) + yaml.safe_dump(cfg, sort_keys=False))
    return CONFIG


def load():
    return yaml.safe_load(CONFIG.read_text())


def check(with_checkpoint=True):
    """[(name, ok, detail)]: the config is v14's with only allowed keys changed; the checkpoint is the recorded one."""
    have = load()
    _, want = build(have["epochs"], have["batch_size"])
    out = [("config_v19.yaml is v14's with only the window, epochs and batch size changed", not diff_keys(have, want),
            str(diff_keys(have, want)) if diff_keys(have, want) else "")]
    if with_checkpoint:
        ok = CHECKPOINT.exists() and sha256_file(CHECKPOINT) == CHECKPOINT_SHA256
        out.append(("pretrained checkpoint present and unchanged", ok,
                    "" if ok else f"{rel(CHECKPOINT)} missing or different (rsync it from the laptop)"))
    out.append(("one-day calendar file present", CALENDAR.exists(), rel(CALENDAR)))
    return out
