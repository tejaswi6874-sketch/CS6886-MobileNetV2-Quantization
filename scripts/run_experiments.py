"""Run the assignment's mixed-precision PTQ sweep from the command line."""

import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_cifar10_loaders, get_calibration_loader
from src.evaluation import load_model, evaluate_quantized
from src.storage import estimate_storage

CONFIGS = [(8, 8), (6, 8), (4, 8), (8, 6), (8, 4), (6, 6), (4, 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/mobilenetv2_cifar10_fp32.pth')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--calibration-batches', type=int, default=100)
    parser.add_argument('--output', default='results/quantization_results_reproduced.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_cifar10_loaders(args.data_root, args.batch_size)
    calibration_loader = get_calibration_loader(args.data_root, args.batch_size)
    fp32_model = load_model(args.checkpoint, device)

    rows = []
    for wb, ab in CONFIGS:
        acc, qmodel = evaluate_quantized(
            fp32_model, test_loader, calibration_loader, device,
            weight_bits=wb, activation_bits=ab,
            calibration_batches=args.calibration_batches,
        )
        storage = estimate_storage(qmodel, weight_bits=wb, activation_bits=ab)
        rows.append({
            'weight_bits': wb,
            'activation_bits': ab,
            'accuracy': round(acc, 4),
            **{k: round(v, 4) for k, v in storage.items()},
        })
        print(f'W{wb}/A{ab}: {acc:.2f}% | {storage["overall_compression_ratio"]:.2f}x')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
