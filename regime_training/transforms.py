"""
Rainfall → model-space transforms
=================================
Scalers that map rainfall (mm per 5 min) into the value space the diffusion
model is trained in, and back.  All of them follow the sklearn scaler
interface (fit / transform / inverse_transform on [N, 1] arrays), so they are
pickled to `scaler.pkl` and used by the generation scripts unchanged.

  standard : sklearn StandardScaler — (x - mean) / std.  The original
             behaviour (v1–v12).
  asinh    : AsinhScaler — z = (asinh(x / s) - mean) / std.

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


def make_scaler(name="standard", asinh_scale=0.035):
    """Build an unfitted scaler from the config's `data_transform` value."""
    if name == "standard":
        return StandardScaler()
    if name == "asinh":
        return AsinhScaler(scale=asinh_scale)
    raise ValueError(f"Unknown data_transform '{name}' (use 'standard' or 'asinh')")
