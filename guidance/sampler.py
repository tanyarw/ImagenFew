"""The guided sampler: regime_training/generate_context.sample_block (E2 settings: known values re-noised,
no RePaint resampling) with one addition, the correction of v14's guess at every network call.

At noise level s, with noisy piece x, context c (model units), mask m (the known first steps) and per-step
weights v:
    guess    d = D(x, s)                         v14's guess of the clean piece
    error    E = sum m * v * (d - c)^2           disagreement on the known steps
    grad     g = dE/dx                           one backward pass through v14 (weights frozen)
    correct  d <- d - w * s^2 * g * (1 - m)      only the unknown part moves; chains not `active` are untouched
With w = 0 the sampler is exactly sample_block (tests/test_guidance.py): same calls, same random draws.
"""
import torch


def guided_denoise(net, x, sigma, labels, ctx, mask, weight, w, active):
    """v14's guess of the clean piece, corrected towards the context. Returns (guess, correction / guess)."""
    if w == 0 or not bool(active.any()):
        with torch.no_grad():
            return net(x, sigma, labels).to(torch.float64), 0.0
    with torch.enable_grad():
        xi = x.detach().requires_grad_(True)
        d = net(xi, sigma, labels).to(torch.float64)
        err = (mask * weight * (d - ctx) ** 2).sum()
        g, = torch.autograd.grad(err, xi)
    a = active.to(d.dtype).view(-1, 1, 1, 1)
    corr = w * sigma ** 2 * g * (1 - mask) * a
    rel = float(corr.abs().mean() / (d.detach().abs().mean() + 1e-12))
    return d.detach() - corr, rel


def sample_piece(net, x_known, mask, labels, t_steps, w=0.0, weight=None, active=None, trace=None):
    """EDM Heun sampling of one piece, known pixels re-noised to the current sigma, guided with strength w."""
    if weight is None:
        weight = torch.ones_like(x_known)
    if active is None:
        active = torch.ones(x_known.shape[0], dtype=torch.bool, device=x_known.device)
    x = torch.randn_like(x_known) * t_steps[0]
    n = len(t_steps) - 1
    for i in range(n):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        known = x_known + t_cur * torch.randn_like(x)
        x = mask * known + (1 - mask) * x
        d, rel = guided_denoise(net, x, t_cur, labels, x_known, mask, weight, w, active)
        if trace is not None:
            trace.append((float(t_cur), rel))
        dx = (x - d) / t_cur
        x_next = x + (t_next - t_cur) * dx
        if i < n - 1:                                       # Heun correction, as in production
            d2, _ = guided_denoise(net, x_next, t_next, labels, x_known, mask, weight, w, active)
            x_next = x + (t_next - t_cur) * (0.5 * dx + 0.5 * (x_next - d2) / t_next)
        x = x_next
    return mask * x_known + (1 - mask) * x


def check_weights(x_known, mask, z_wet, variant, wet_weight):
    """Per-step weights v and per-chain switch for the three versions of the check.

    plain  : every known step weight 1, every chain guided
    wet    : wet known steps weigh `wet_weight` times the dry ones, every chain guided
    rainy  : as wet, but only chains whose known steps hold rain are guided
    The weights are rescaled so their mean over each chain's known steps is 1: they shift the balance between
    wet and dry steps without changing the overall size of the correction, so a strength w means the same in
    every version.
    """
    wet = (x_known > z_wet) & (mask > 0)
    if variant == "plain":
        weight = torch.ones_like(x_known)
    elif variant in ("wet", "rainy"):
        weight = 1.0 + (wet_weight - 1.0) * wet.to(x_known.dtype)
        n_known = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        mean_known = (weight * mask).flatten(1).sum(dim=1) / n_known
        weight = weight / torch.where(mean_known > 0, mean_known, torch.ones_like(mean_known)).view(-1, 1, 1, 1)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "rainy":
        active = wet.flatten(1).any(dim=1)
    else:
        active = torch.ones(x_known.shape[0], dtype=torch.bool, device=x_known.device)
    return weight, active
