"""
Rainfall → model-space transforms
=================================
Scalers that map rainfall (mm per 5 min) into the value space the diffusion
model is trained in, and back.  All of them follow the sklearn scaler
interface (fit / transform / inverse_transform on [N, 1] arrays), so they are
pickled to `scaler.pkl` and used by the generation scripts unchanged.

  standard : sklearn StandardScaler — (x - mean) / std.  The original
             behaviour (v1–v12).
  log1p    : Log1pScaler — z = (log(1 + x) - mean) / std.  (v13)
  asinh    : AsinhScaler — z = (asinh(x / s) - mean) / std.  (v14)

Optional `data_scale` (config key, default 1.0): multiplies z by a constant
(ScaledScaler, v18).  0.5 matches ImagenFew's hard-coded EDM sigma_data = 0.5.

Why asinh
---------
asinh(x / s) is ~linear for x << s and ~log(2x / s) for x >> s.  So light rain
keeps its resolution while storm peaks are compressed: on the 2000–2007
training split the largest value drops from z ≈ 118 (standard) to z ≈ 14
(asinh, s = 0.035).  Standardising afterwards keeps mean 0 / std 1, exactly as
before, so the ONLY thing that changes between the two runs is the shape of
the mapping — the EDM settings (sigma_data = 0.5 etc.) see the same overall
scale.

Pure numpy + sklearn (no torch), so local analysis scripts can import it.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class AsinhScaler:
    """z = (asinh(x / s) - mean) / std, with mean / std fitted on asinh(x / s)."""

    def __init__(self, scale=0.035):
        # s in mm per 5 min.  Default = median of wet steps (>= 0.005 mm)
        # on the 2000–2007 training split.
        self.scale = float(scale)
        self._std = StandardScaler()

    def fit(self, x):
        self._std.fit(np.arcsinh(np.asarray(x, dtype=np.float64) / self.scale))
        return self

    def transform(self, x):
        a = np.arcsinh(np.asarray(x, dtype=np.float64) / self.scale)
        return self._std.transform(a).astype(np.float32)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, z):
        a = self._std.inverse_transform(np.asarray(z, dtype=np.float64))
        return (self.scale * np.sinh(a)).astype(np.float32)

    def __repr__(self):
        return f"AsinhScaler(scale={self.scale})"


class Log1pScaler:
    """z = (log(1 + x) - mean) / std, with mean / std fitted on log(1 + x).

    Unlike asinh there is no scale parameter, so with x in mm per 5 min the
    map is ~linear up to ~1 mm and only compresses the storm peaks.
    """

    def __init__(self):
        self._std = StandardScaler()

    def fit(self, x):
        self._std.fit(np.log1p(np.asarray(x, dtype=np.float64)))
        return self

    def transform(self, x):
        a = np.log1p(np.asarray(x, dtype=np.float64))
        return self._std.transform(a).astype(np.float32)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, z):
        a = self._std.inverse_transform(np.asarray(z, dtype=np.float64))
        return np.expm1(a).astype(np.float32)

    def __repr__(self):
        return "Log1pScaler()"


class ScaledScaler:
    """z = factor * base(x): any scaler above, then multiplied by a constant.  (v18)

    Why: EDM (Karras et al. 2022) assumes the clean data have standard deviation
    sigma_data, and ImagenFew hard-codes sigma_data = 0.5 (models/ImagenFew/ImagenFew.py,
    networks.py EDMPrecond).  Every scaler above standardises to std 1, so the
    preconditioning and the loss weight are set for data half as spread out as ours
    (code_plan/LOSS_FUNCTION.md, finding 4).  factor = 0.5 brings the data to std 0.5,
    which matches sigma_data AND keeps every other EDM setting (P_mean, P_std,
    sigma_min, sigma_max) at the values they were tuned for, and keeps the pretrained
    checkpoint's preconditioning meaning what it meant in pretraining.  Setting
    sigma_data = 1 instead would also match, but would shift the training noise
    distribution relative to the data.
    """

    def __init__(self, base, factor):
        if not factor > 0:
            raise ValueError(f"data_scale must be positive, got {factor}")
        self.base = base
        self.factor = float(factor)

    def fit(self, x):
        self.base.fit(x)
        return self

    def transform(self, x):
        return (np.asarray(self.base.transform(x), dtype=np.float64) * self.factor).astype(np.float32)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, z):
        return np.asarray(self.base.inverse_transform(np.asarray(z, dtype=np.float64) / self.factor),
                          dtype=np.float32)

    def __repr__(self):
        return f"ScaledScaler({self.base!r}, factor={self.factor})"


def make_scaler(name="standard", asinh_scale=0.035, data_scale=1.0):
    """Build an unfitted scaler from the config's `data_transform` (and optional
    `data_scale`) values.  data_scale = 1.0 (the default) returns the plain scaler,
    so every config without the key builds exactly what it built before."""
    if name == "standard":
        base = StandardScaler()
    elif name == "log1p":
        base = Log1pScaler()
    elif name == "asinh":
        base = AsinhScaler(scale=asinh_scale)
    else:
        raise ValueError(f"Unknown data_transform '{name}' (use 'standard', 'log1p' or 'asinh')")
    return base if float(data_scale) == 1.0 else ScaledScaler(base, data_scale)
