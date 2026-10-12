"""Occurrence-matched latent Gaussian AR (scripts/baselines/run_copula_arima_occ.py, plain mode) for any site.

The fit and the generator are the existing script's, called through its own functions (EmpiricalQuantile,
both_above, levinson_durbin); what this adapter adds is the site's wet threshold, any training years, and an
outage mask (joint wet probabilities are counted only over pairs of observed steps; the quantile table
uses observed steps only). With no mask, the output is identical to the existing script's
(tests/test_baseline_regression.py).

The storm-and-cell baseline is not adapted: its 0.005 mm cut-off sits inside the simulator and its parameter
search box was set for Astlingen, so a site-generic version would change scripts/baselines/run_bartlett_lewis.py.
"""
import importlib.util
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter

from .provenance import ROOT
from .timegrid import STEPS_PER_YEAR

_BASE = ROOT / "scripts" / "baselines"


def _module(name):
    if name in sys.modules:
        return sys.modules[name]
    if str(_BASE) not in sys.path:
        sys.path.insert(0, str(_BASE))
    spec = importlib.util.spec_from_file_location(name, _BASE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def joint_wet(wet, k, observed=None):
    """P(wet_t and wet_t+k), over pairs where both steps are observed."""
    pair = wet[:-k] & wet[k:]
    if observed is None:
        return float(np.mean(pair))
    both = observed[:-k] & observed[k:]
    return float(np.mean(pair[both]))


def latent_rho(q, thresholds, weights, both_above):
    """Latent correlation reproducing each joint wet probability (run_copula_arima_occ.latent_targets)."""
    rho = np.zeros(len(q))
    for i, qk in enumerate(q):
        f = lambda r: sum(w * both_above(c, r) for c, w in zip(thresholds, weights)) - qk  # noqa: E731
        if f(0.0) >= 0:
            continue
        rho[i] = brentq(f, 0.0, 1.0 - 1e-9, xtol=1e-10)
    return rho


def fit(x, tau, p=64, mask=None):
    occ = _module("run_copula_arima_occ")
    obs = None if mask is None or not np.any(mask) else ~np.asarray(mask, bool)
    eq = occ.EmpiricalQuantile(x if obs is None else x[obs], wet_threshold=tau)
    wet = x >= tau
    q = np.array([joint_wet(wet, k, obs) for k in range(1, p + 1)])
    rho = latent_rho(q, [eq.z_threshold], [1.0], occ.both_above)
    r = np.r_[1.0, rho]
    T = r[np.abs(np.subtract.outer(np.arange(p + 1), np.arange(p + 1)))]
    min_eig = float(np.linalg.eigvalsh(T).min())
    if min_eig <= 0:
        raise ValueError("latent correlation targets are not positive definite")
    phi, sigma2 = occ.levinson_durbin(r, p)
    return dict(eq=eq, phi=phi, sigma=float(np.sqrt(sigma2)), q=q, rho=rho, min_eig=min_eig, p=p)


def generate(model, years, seed):
    """Identical random stream to run_copula_arima_occ.main (np.random.seed, 2,000-step burn-in)."""
    n = years * STEPS_PER_YEAR
    burn = 2000
    np.random.seed(seed)
    eps = np.random.normal(0, model["sigma"], n + burn)
    z = lfilter([1.0], np.r_[1.0, -model["phi"]], eps)[burn:]
    z = (z - np.mean(z)) / (np.std(z) + 1e-9)
    return model["eq"].z_to_x(z)
