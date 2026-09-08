"""Evaluate the saved CIFAR-10 MobileNetV2 checkpoint at a chosen bit-width."""

import argparse
import copy
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model import build_model
from src.quantization import replace_modules, calibrate


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            output = model(images.to(device))
            correct += (output.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/mobilenetv2_cifar10_fp32.pth")
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    checkpoint = torch.load(os.path.join(ROOT, args.checkpoint), map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)
    calibration_loader = DataLoader(train, batch_size=128, shuffle=False, num_workers=0)

    fp32_accuracy = evaluate(model, test_loader, device)
    quantized = replace_modules(copy.deepcopy(model), args.weight_bits, args.activation_bits, True)
    quantized = calibrate(quantized.to(device), calibration_loader, device, args.calibration_batches)
    accuracy = evaluate(quantized, test_loader, device)

    print(f"Device: {device}")
    print(f"Configuration: W{args.weight_bits}/A{args.activation_bits}, per-channel weights")
    print(f"FP32 accuracy: {fp32_accuracy:.2f}%")
    print(f"Quantized accuracy: {accuracy:.2f}%")
    print(f"Accuracy drop: {fp32_accuracy - accuracy:.2f} percentage points")
    print(f"Weight compression ratio: {32 / args.weight_bits:.2f}x")
    print(f"Activation compression ratio: {32 / args.activation_bits:.2f}x")


if __name__ == "__main__":
    main()
