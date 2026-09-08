"""Configurable fake-quantization modules for MobileNetV2 PTQ."""

import torch
from torch import nn
import torch.nn.functional as F


def qrange(bits: int, symmetric: bool):
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if symmetric else (0, 2 ** bits - 1)


def qparams(x, bits=8, symmetric=True, per_channel=False):
    qmin, qmax = qrange(bits, symmetric)
    x = x.detach()
    if per_channel:
        dims = tuple(range(1, x.ndim))
        xmin, xmax = x.amin(dims, keepdim=True), x.amax(dims, keepdim=True)
    else:
        xmin, xmax = x.min(), x.max()
    if symmetric:
        scale = torch.maximum(xmin.abs(), xmax.abs()) / float(qmax)
        zero = torch.zeros_like(scale)
    else:
        scale = (xmax - xmin) / float(qmax - qmin)
        zero = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return scale, zero, qmin, qmax


def fake_quant(x, scale, zero, qmin, qmax):
    q = torch.round(x / scale + zero).clamp(qmin, qmax)
    return (q - zero) * scale


class QuantizedConv2d(nn.Module):
    def __init__(self, layer, bits=8, per_channel=True):
        super().__init__()
        self.layer, self.bits = layer, bits
        scale, zero, self.qmin, self.qmax = qparams(layer.weight, bits, True, per_channel)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero)

    def forward(self, x):
        w = fake_quant(self.layer.weight, self.scale, self.zero_point, self.qmin, self.qmax)
        return F.conv2d(x, w, self.layer.bias, self.layer.stride, self.layer.padding,
                        self.layer.dilation, self.layer.groups)


class QuantizedLinear(nn.Module):
    def __init__(self, layer, bits=8, per_channel=True):
        super().__init__()
        self.layer, self.bits = layer, bits
        scale, zero, self.qmin, self.qmax = qparams(layer.weight, bits, True, per_channel)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero)

    def forward(self, x):
        w = fake_quant(self.layer.weight, self.scale, self.zero_point, self.qmin, self.qmax)
        return F.linear(x, w, self.layer.bias)


class ActivationFakeQuant(nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.bits, self.calibrating, self.frozen = bits, True, False
        self.register_buffer("min_value", torch.tensor(float("inf")))
        self.register_buffer("max_value", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))

    def forward(self, x):
        if self.calibrating:
            self.min_value.copy_(torch.minimum(self.min_value, x.detach().min()))
            self.max_value.copy_(torch.maximum(self.max_value, x.detach().max()))
        if not self.frozen:
            return x
        qmin, qmax = qrange(self.bits, False)
        return fake_quant(x, self.scale, self.zero_point, qmin, qmax)

    def freeze(self):
        scale, zero, _, _ = qparams(torch.stack([self.min_value, self.max_value]), self.bits, False)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero)
        self.calibrating, self.frozen = False, True


def replace_modules(model, weight_bits=8, activation_bits=8, weight_per_channel=True):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(model, name, QuantizedConv2d(child, weight_bits, weight_per_channel))
        elif isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear(child, weight_bits, weight_per_channel))
        elif isinstance(child, (nn.ReLU, nn.ReLU6)):
            setattr(model, name, nn.Sequential(child, ActivationFakeQuant(activation_bits)))
        else:
            replace_modules(child, weight_bits, activation_bits, weight_per_channel)
    return model


def calibrate(model, loader, device, batches=100):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            model(images.to(device))
            if i + 1 >= batches:
                break
    for module in model.modules():
        if isinstance(module, ActivationFakeQuant):
            module.freeze()
    return model
