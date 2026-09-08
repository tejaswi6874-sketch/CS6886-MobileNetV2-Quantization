"""MobileNetV2 model construction utilities."""

from torch import nn
from torchvision.models import mobilenet_v2


def build_model(num_classes: int = 10) -> nn.Module:
    """Build the MobileNetV2 architecture used in the assignment."""
    model = mobilenet_v2(num_classes=num_classes)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
