"""Evaluation helpers for FP32 and fake-quantized MobileNetV2."""

import copy

import torch

from .model import build_model, evaluate_accuracy
from .quantization import replace_modules, calibrate


def load_model(checkpoint_path, device, num_classes=10):
    """Load the assignment's FP32 checkpoint."""
    model = build_model(num_classes).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def quantize_and_calibrate(fp32_model, calibration_loader, device,
                           weight_bits=8, activation_bits=8,
                           calibration_batches=100):
    """Create an independent fake-quantized copy and calibrate activation ranges."""
    model = copy.deepcopy(fp32_model)
    model = replace_modules(
        model,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        weight_per_channel=True,
    ).to(device)
    return calibrate(model, calibration_loader, device, batches=calibration_batches)


def evaluate_quantized(fp32_model, test_loader, calibration_loader, device,
                       weight_bits=8, activation_bits=8,
                       calibration_batches=100):
    """Quantize, calibrate and evaluate one independent configuration."""
    model = quantize_and_calibrate(
        fp32_model, calibration_loader, device,
        weight_bits, activation_bits, calibration_batches,
    )
    return evaluate_accuracy(model, test_loader, device), model
