"""Logical storage estimates for mixed-precision fake quantization."""

import torch
from torch import nn


def count_activation_sites(model):
    return sum(isinstance(m, nn.Sequential) and len(m) == 2 and m[1].__class__.__name__ == "ActivationFakeQuant" for m in model.modules())


def estimate_storage(model, weight_bits=8, activation_bits=8):
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    quant_weights = sum(m.layer.weight.numel() for m in model.modules() if hasattr(m, "layer") and isinstance(m.layer, (nn.Conv2d, nn.Linear)))
    scales = sum(m.scale.numel() for m in model.modules() if hasattr(m, "scale") and hasattr(m, "layer"))
    fp32_mb = (params + buffers) * 4 / 1024**2
    quant_mb = quant_weights * weight_bits / 8 / 1024**2
    scale_mb = scales * 4 / 1024**2
    remaining_mb = max(params - quant_weights, 0) * 4 / 1024**2
    buffer_mb = buffers * 4 / 1024**2
    metadata_mb = count_activation_sites(model) * 8 / 1024**2
    compressed_mb = quant_mb + scale_mb + remaining_mb + buffer_mb + metadata_mb
    return {"fp32_size_mb": fp32_mb, "compressed_size_mb": compressed_mb,
            "weight_compression_ratio": 32 / weight_bits,
            "activation_compression_ratio": 32 / activation_bits,
            "overall_compression_ratio": fp32_mb / compressed_mb}
