"""Run records: every command writes one JSON file saying what ran, on which code, with which inputs.

A record holds the command and its parameters, the git commit and whether tracked files had uncommitted
changes (with a hash of the diff), library versions, the host and SLURM job, SHA-256 hashes of inputs and
outputs, start and end times, and the status (with the traceback if the command failed).
"""
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state():
    commit = (_git("rev-parse", "HEAD") or "").strip() or None
    diff = _git("diff", "HEAD", "--", ".") or ""
    return {"commit": commit, "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
            "dirty": bool(diff.strip()), "diff_sha256": sha256_bytes(diff.encode()) if diff.strip() else None}


def _versions():
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "hmmlearn", "torch", "yaml"):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, "__version__", "?")
    return out


def rel(path):
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


class RunRecord:
    """Context manager: `with RunRecord('labels', site, params, out_path) as rec: rec.input(p); rec.output(p)`."""

    def __init__(self, command, site_name, params, path):
        self.path = Path(path)
        self.data = {"command": command, "site": site_name, "params": params, "argv": sys.argv,
                     "git": git_state(), "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "started": _dt.datetime.now().isoformat(timespec="seconds"), "inputs": {}, "outputs": {},
                     "results": {}}

    def input(self, path, hash_it=True):
        self.data["inputs"][rel(path)] = sha256_file(path) if hash_it else None

    def output(self, path):
        self.data["outputs"][rel(path)] = sha256_file(path)

    def result(self, **kv):
        self.data["results"].update(kv)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.data["versions"] = _versions()
        self.data["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.data["status"] = "ok" if exc_type is None else "failed"
        if exc_type is not None:
            self.data["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, default=str) + "\n")
        return False


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    tmp.replace(path)                       # atomic: a reader never sees a half-written file
    return path


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return rel(o)
    return str(o)
