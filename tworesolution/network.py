"""The fine model's network: v14's U-Net with the half-hour totals as an extra input.

Why not simply a second input channel: DhariwalUNet's first layer (DynamicConv2d) resizes one stored weight bank
to however many input channels it is given, so with two channels the weights applied to the rain channel would
change too, and the network would no longer be v14. The totals therefore enter through their own 3x3 convolution,
initialised to zero, whose output is added to the first layer's output (a "zero convolution", as in ControlNet,
Zhang et al. 2023). While that layer is zero the network is v14 exactly (tests/test_tworesolution.py).

The condition is passed to EDMPrecond as a keyword argument, which EDMPrecond hands to the U-Net unchanged: unlike
the noisy rain image it is not multiplied by c_in = 1 / sqrt(sigma^2 + sigma_data^2), so the totals reach the
network at full size at every noise level.
"""
import argparse

import torch
from omegaconf import OmegaConf
from torch.nn.functional import silu

from models.ImagenFew.ImagenFew import ImagenFew
from models.ImagenFew.networks import Conv2d, UNetBlock


class ConditionedUNet(torch.nn.Module):
    """A DhariwalUNet plus a zero-initialised input path for the condition image [B, 1, 8, 8]."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        first = next(iter(unet.enc.values()))
        self.cond_conv = Conv2d(1, first.out_channels, kernel=3, init_mode="kaiming_uniform", init_weight=0,
                                init_bias=0)

    def forward(self, x, noise_labels, class_labels, cond, augment_labels=None):
        # DhariwalUNet.forward (models/ImagenFew/networks.py), with one marked addition.
        u = self.unet
        C = x.shape[1]
        emb = u.map_noise(noise_labels)
        if u.map_augment is not None and augment_labels is not None:
            emb = emb + u.map_augment(augment_labels)
        emb = silu(u.map_layer0(emb))
        emb = u.map_layer1(emb)
        if u.map_label is not None:
            tmp = class_labels
            if u.training and u.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= u.label_dropout).to(tmp.dtype)
            emb = emb + u.map_label(tmp)
        emb = silu(emb)

        skips = []
        for i, block in enumerate(u.enc.values()):
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            if i == 0:
                x = x + self.cond_conv(cond.to(x.dtype))          # the addition: the condition joins here
            skips.append(x)
        for block in u.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        return u.out_conv(silu(u.out_norm(x)), C)


def load_args(config_path, device):
    args = argparse.Namespace(**OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    args.device = device
    return args


def attach_condition(model):
    """Give an ImagenFew model the condition input; its EMA is rebuilt over the new set of parameters."""
    device = next(model.parameters()).device
    model.net.model = ConditionedUNet(model.net.model).to(device)
    if model.use_ema:
        model.reset_ema()
    return model


def load_plain(args, ckpt_path, device):
    """An ImagenFew model with a trained checkpoint, loaded strictly (every key must match)."""
    model = ImagenFew(args, device).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    if model.use_ema:
        model.model_ema.load_state_dict(ck["ema_model"], strict=True)
    return model.eval()


def fine_from_v14(args, v14_ckpt, device):
    """The fine model's starting point: v14's EMA weights (the ones v14 generates with) + the zero condition layer."""
    model = load_plain(args, v14_ckpt, device)
    if model.use_ema:
        model.model_ema.copy_to(model.net)
    return attach_condition(model)


def save_fine(model, path, **meta):
    torch.save({"model": model.state_dict(), "ema_model": model.model_ema.state_dict(), **meta}, path)


def load_fine(args, ckpt_path, device):
    model = attach_condition(ImagenFew(args, device).to(device))
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.model_ema.load_state_dict(ck["ema_model"], strict=True)
    return model.eval()
