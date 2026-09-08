"""Logical persistent-storage estimates for mixed-precision fake quantization."""

from torch import nn


def count_quantized_weight_values(model):
    return sum(
        m.layer.weight.numel()
        for m in model.modules()
        if hasattr(m, "layer") and isinstance(m.layer, (nn.Conv2d, nn.Linear))
    )


def count_weight_scale_values(model):
    return sum(
        m.scale.numel()
        for m in model.modules()
        if hasattr(m, "scale") and hasattr(m, "layer")
    )


def count_activation_sites(model):
    return sum(
        isinstance(m, nn.Sequential)
        and len(m) == 2
        and m[1].__class__.__name__ == "ActivationFakeQuant"
        for m in model.modules()
    )


def count_non_quantization_buffer_values(model):
    """Count persistent FP32 buffers such as BatchNorm statistics only."""
    excluded = {"scale", "zero_point", "min_value", "max_value"}
    return sum(
        b.numel()
        for module in model.modules()
        for name, b in module._buffers.items()
        if b is not None and name not in excluded
    )


def estimate_storage(model, weight_bits=8, activation_bits=8):
    """Estimate logical packed storage, matching the report's accounting assumptions.

    Quantized Conv2d/Linear weights are bit-packed. Per-channel weight scales and
    activation scale/zero-point metadata are stored as FP32. Biases, BatchNorm
    parameters, and non-quantization buffers remain FP32. Quantization buffers
    used only to hold metadata are not double-counted as ordinary FP32 buffers.
    """
    all_param_values = sum(p.numel() for p in model.parameters())
    quant_weight_values = count_quantized_weight_values(model)
    weight_scale_values = count_weight_scale_values(model)
    other_fp32_param_values = max(all_param_values - quant_weight_values, 0)
    non_quant_buffer_values = count_non_quantization_buffer_values(model)
    activation_sites = count_activation_sites(model)

    fp32_mb = (all_param_values + non_quant_buffer_values) * 4 / 1024**2
    quantized_weight_mb = quant_weight_values * weight_bits / 8 / 1024**2
    weight_scale_mb = weight_scale_values * 4 / 1024**2
    remaining_parameters_mb = other_fp32_param_values * 4 / 1024**2
    fp32_buffers_mb = non_quant_buffer_values * 4 / 1024**2
    activation_metadata_mb = activation_sites * 8 / 1024**2
    compressed_mb = (
        quantized_weight_mb
        + weight_scale_mb
        + remaining_parameters_mb
        + fp32_buffers_mb
        + activation_metadata_mb
    )

    return {
        "fp32_size_mb": fp32_mb,
        "quantized_weight_mb": quantized_weight_mb,
        "weight_scale_mb": weight_scale_mb,
        "remaining_parameters_mb": remaining_parameters_mb,
        "fp32_buffers_mb": fp32_buffers_mb,
        "activation_metadata_mb": activation_metadata_mb,
        "activation_sites": activation_sites,
        "compressed_size_mb": compressed_mb,
        "weight_compression_ratio": 32 / weight_bits,
        "activation_compression_ratio": 32 / activation_bits,
        "overall_compression_ratio": fp32_mb / compressed_mb,
    }
