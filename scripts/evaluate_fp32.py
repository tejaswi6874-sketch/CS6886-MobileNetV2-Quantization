"""Evaluate a trained FP32 MobileNetV2 checkpoint on CIFAR-10."""

import argparse
import torch

from src.data import get_cifar10_loaders
from src.model import load_fp32_model, evaluate_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/mobilenetv2_cifar10_fp32.pth')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_cifar10_loaders(args.data_root, args.batch_size)
    model = load_fp32_model(args.checkpoint, device)
    acc = evaluate_accuracy(model, test_loader, device)
    print(f'device={device}')
    print(f'top1_accuracy={acc:.2f}%')


if __name__ == '__main__':
    main()
