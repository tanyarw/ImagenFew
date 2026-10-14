"""Generation at both levels with the production sampler.

`heun` is DiffusionProcess.impute (models/ImagenFew/sampler.py) as every production run used it: nothing known,
no churn, 36 Heun steps from sigma 80 to 0.002 (rho 7), the last step Euler. Given the same starting noise the
output is identical (tests/test_tworesolution.py); the only addition is the optional condition.

Coarse: independent 64-step pieces (32 h) joined end to end on the calendar timeline, no smoothing (as the
window pilot); each piece gets the majority seasonal phase of its 32 hours. Fine: one window per 10 half-hours
(grid.generation_starts), each given the totals of its own and its neighbours' half-hours; windows are
independent given the coarse series, so nothing is chained. Member m's starting noise comes from seed + m and
does not depend on the batch size.
"""
import numpy as np
import torch

from .grid import L, N_CLASSES, clean, condition_windows, generation_starts, kept, window_labels
from .mass import finish

SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7          # DiffusionProcess


def schedule(net, num_steps, device):
    smin, smax = max(SIGMA_MIN, net.sigma_min), min(SIGMA_MAX, net.sigma_max)
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    t = (smax ** (1 / RHO) + idx / (num_steps - 1) * (smin ** (1 / RHO) - smax ** (1 / RHO))) ** RHO
    return torch.cat([net.round_sigma(t), torch.zeros_like(t[:1])])


@torch.no_grad()
def heun(net, xT, labels, t_steps, cond=None):
    kw = {} if cond is None else {"cond": cond}
    x = xT.to(torch.float64) * t_steps[0]
    n = len(t_steps) - 1
    for i in range(n):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        d = (x - net(x, t_cur, labels, **kw).to(torch.float64)) / t_cur
        x_next = x + (t_next - t_cur) * d
        if i < n - 1:
            d2 = (x_next - net(x_next, t_next, labels, **kw).to(torch.float64)) / t_next
            x_next = x + (t_next - t_cur) * (0.5 * d + 0.5 * d2)
        x = x_next
    return x


def _noise(n, seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, 1, 8, 8, generator=g)


def _to_mm(scaler, z):
    z = np.asarray(z, dtype=np.float64)
    return scaler.inverse_transform(z.reshape(-1, 1)).reshape(z.shape).astype(np.float64)


def _run(model, labels, xT, cond_mm, scaler, batch, device, num_steps):
    """Sample windows in batches; returns [n, L] in mm."""
    net = model.net
    t_steps = schedule(net, num_steps, device)
    eye = torch.eye(N_CLASSES, device=device)
    model.ts_to_img(torch.zeros(1, L, 1))        # DelayEmbedder.img_to_ts reads the shape cached by ts_to_img
    out = []
    with model.ema_scope():
        for i in range(0, len(labels), batch):
            lab = eye[torch.as_tensor(labels[i:i + batch], device=device)]
            cond = None
            if cond_mm is not None:
                c = torch.as_tensor(scaler.transform(cond_mm[i:i + batch].reshape(-1, 1)).reshape(-1, L, 1),
                                    dtype=torch.float32, device=device)
                cond = model.ts_to_img(c)
            img = heun(net, xT[i:i + batch].to(device), lab, t_steps, cond=cond)
            out.append(model.img_to_ts(img.to(torch.float32))[:, :, 0].cpu().numpy())
    return _to_mm(scaler, np.concatenate(out))


def generate_coarse(model, scaler, profile30, n30, seed, tau30, batch=2048, device="cuda", num_steps=36):
    """One coarse series of n30 half-hours (mm per half-hour, cleaned with the coarse threshold)."""
    starts = L * np.arange(-(-n30 // L))
    labels = window_labels(profile30, starts)
    x = _run(model, labels, _noise(len(starts), seed), None, scaler, batch, device, num_steps)
    return clean(x.reshape(-1)[:n30], tau30)


def disaggregate(model, scaler, profile5, x30, seed, tau5, batch=2048, device="cuda", num_steps=36):
    """(5-minute series, mass-matching counts) for a series of half-hour totals x30 starting on 1 January."""
    x30 = np.asarray(x30, dtype=np.float64)
    starts = generation_starts(len(x30))
    labels = window_labels(profile5, starts)
    raw = _run(model, labels, _noise(len(starts), seed), condition_windows(x30, starts), scaler, batch, device,
               num_steps)
    fine, counts = finish(kept(raw, len(x30)), x30, tau5)
    return fine.reshape(-1), counts
