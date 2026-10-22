"""Keeps the test year single-use (code_plan/TEST_YEAR_PLAN.md section 5). Standard library only.

`unlock` runs every check and returns the only object the 2009 loaders accept. Each check raises `Locked` with
the reason; nothing is written before all of them pass, and the claim file is the last step."""
import datetime as dt
import hashlib
import json
import os
import subprocess

from . import config as K


class Locked(SystemExit):
    """The test year stays closed. A SystemExit, so the command line prints the reason and exits non-zero."""

    def __init__(self, reason):
        super().__init__(f"test year {K.TEST_YEAR} stays closed: {reason}")


class Key:
    """Proof that every check passed. Made only by `unlock`."""

    def __init__(self, prereg_sha256, _token):
        if _token is not _TOKEN:
            raise Locked("a Key is made only by guard.unlock")
        self.prereg_sha256 = prereg_sha256


_TOKEN = object()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(["git", *args], cwd=K.ROOT, capture_output=True, text=True)


def committed(paths):
    """[problems]: each path tracked by git with no uncommitted change."""
    out = []
    for p in paths:
        rel = os.path.relpath(p, K.ROOT)
        if _git("ls-files", "--error-unmatch", rel).returncode != 0:
            out.append(f"{rel} is not committed")
        elif _git("status", "--porcelain", "--", rel).stdout.strip():
            out.append(f"{rel} has uncommitted changes")
    return out


def load_prereg():
    if not K.PREREG.exists():
        raise Locked(f"no preregistration ({os.path.relpath(K.PREREG, K.ROOT)}); run `python -m testyear register`")
    return json.loads(K.PREREG.read_text()), sha256_file(K.PREREG)


def series_unchanged(prereg):
    """[problems]: every registered series file is present with its registered hash."""
    out = []
    for name, sha in prereg["series_sha256"].items():
        path = K.series_csv(name)
        if not path.exists():
            out.append(f"{path.name} missing")
        elif sha256_file(path) != sha:
            out.append(f"{path.name} changed since registration")
    return out


def rehearsal_ok(prereg_sha):
    path = K.result_path(K.REHEARSAL_YEAR)
    if not path.exists():
        return [f"no rehearsal ({path.name}); run `python -m testyear rehearse`"]
    r = json.loads(path.read_text())
    out = []
    if r.get("prereg_sha256") != prereg_sha:
        out.append(f"{path.name} was made from another preregistration")
    if not r.get("checks", {}).get("sewer_reproduces_flood_control", False):
        out.append(f"{path.name}: the real-year sewer run did not reproduce flood-control's saved {K.REHEARSAL_YEAR}")
    return out


def claim(path, prereg_sha):
    """Create `path` exclusively: a second or concurrent run fails here. Never removed by code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise Locked(f"{path.name} exists: the test year was already opened (if that run crashed before writing "
                     f"its result, inspect the claim and remove it by hand, recording why in the diary)")
    with os.fdopen(fd, "w") as f:
        json.dump({"opened": dt.datetime.now().isoformat(timespec="seconds"), "prereg_sha256": prereg_sha,
                   "git_commit": _git("rev-parse", "HEAD").stdout.strip()}, f, indent=1)


def unlock(spend_test_year):
    """Every check, then the claim. Returns (prereg, Key)."""
    if not spend_test_year:
        raise Locked("pass --spend-test-year: the test year is opened once, for the registered generator only")
    if K.result_path(K.TEST_YEAR).exists():
        raise Locked(f"{K.result_path(K.TEST_YEAR).name} exists; the result is final")
    prereg, sha = load_prereg()
    problems = committed([K.PREREG, K.PACKAGE]) + series_unchanged(prereg) + rehearsal_ok(sha)
    if problems:
        raise Locked("; ".join(problems))
    claim(K.claim_path(K.TEST_YEAR), sha)
    return prereg, Key(sha, _TOKEN)


def require(year, key):
    """Loaders call this before reading a real year: the test year needs a Key."""
    if year == K.TEST_YEAR and not isinstance(key, Key):
        raise Locked(f"reading {year} needs the Key from guard.unlock")
