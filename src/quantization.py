"""Core uniform post-training fake-quantization for MobileNetV2."""

import torch
from torch import nn
import torch.nn.functional as F


def qrange(bits: int, symmetric: bool):
    if bits < 2:
        raise ValueError("bits must be >= 2")
    if symmetric:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def qparams(x, bits=8, symmetric=True, per_channel=False):
    """Calculate uniform linear quantization parameters."""
    qmin, qmax = qrange(bits, symmetric)
    x = x.detach()

    if per_channel:
        dims = tuple(range(1, x.ndim))
        xmin = x.amin(dims, keepdim=True)
        xmax = x.amax(dims, keepdim=True)
    else:
        xmin = x.min()
        xmax = x.max()

    if symmetric:
        scale = torch.maximum(xmin.abs(), xmax.abs()) / float(qmax)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.zeros_like(scale)
    else:
        scale = (xmax - xmin) / float(qmax - qmin)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.round(qmin - xmin / scale).clamp(qmin, qmax)

    return scale, zero, qmin, qmax


def fake_quant(x, scale, zero, qmin, qmax):
    """Quantize to integer levels and immediately dequantize to FP32."""
    q = torch.round(x / scale + zero).clamp(qmin, qmax)
    return (q - zero) * scale


class QuantizedConv2d(nn.Module):
    """Conv2d with symmetric weight fake quantization."""

    def __init__(self, layer, bits=8, per_channel=True):
        super().__init__()
        self.layer = layer
        self.bits = bits
        self.per_channel = per_channel
        scale, zero, self.qmin, self.qmax = qparams(
            layer.weight, bits, symmetric=True, per_channel=per_channel
        )
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero)

    def forward(self, x):
        w = fake_quant(self.layer.weight, self.scale, self.zero_point, self.qmin, self.qmax)
        return F.conv2d(
            x, w, self.layer.bias, self.layer.stride, self.layer.padding,
            self.layer.dilation, self.layer.groups
        )


class QuantizedLinear(nn.Module):
    """Linear layer with symmetric weight fake quantization."""

    def __init__(self, layer, bits=8, per_channel=True):
        super().__init__()
        self.layer = layer
        self.bits = bits
        self.per_channel = per_channel
        scale, zero, self.qmin, self.qmax = qparams(
            layer.weight, bits, symmetric=True, per_channel=per_channel
        )
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero)

    def forward(self, x):
        w = fake_quant(self.layer.weight, self.scale, self.zero_point, self.qmin, self.qmax)
        return F.linear(x, w, self.layer.bias)


class ActivationFakeQuant(nn.Module):
    """Data-calibrated asymmetric per-tensor activation fake quantizer."""

    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.calibrating = True
        self.frozen = False
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
        qmin, qmax = qrange(self.bits, symmetric=False)
        return fake_quant(x, self.scale, self.zero_point, qmin, qmax)

    def freeze(self):
        observed_range = torch.stack((self.min_value, self.max_value))
        scale, zero, _, _ = qparams(observed_range, self.bits, symmetric=False)
        self.scale.copy_(scale.reshape_as(self.scale))
        self.zero_point.copy_(zero.reshape_as(self.zero_point))
        self.calibrating = False
        self.frozen = True


def replace_modules(model, weight_bits=8, activation_bits=8, weight_per_channel=True):
    """Recursively replace Conv2d/Linear and ReLU/ReLU6 modules."""
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
    """Collect activation ranges and then freeze the calibrated parameters."""
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
