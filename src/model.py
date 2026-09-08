"""MobileNetV2 model construction and evaluation helpers for CIFAR-10."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v2
from PIL import Image


def build_model(num_classes: int = 10) -> nn.Module:
    """Build the MobileNetV2 architecture used by the assignment."""
    model = mobilenet_v2(num_classes=num_classes)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_fp32_model(checkpoint_path: str, device: torch.device, num_classes: int = 10) -> nn.Module:
    """Load the trained FP32 checkpoint."""
    model = build_model(num_classes).to(device)
    state = torch.load(Path(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def evaluate_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    """Evaluate top-1 accuracy on a dataloader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / total


def prepare_image(path: str, transform) -> torch.Tensor:
    """Load one image and return a batched tensor using the evaluation transform."""
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)
