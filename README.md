# CS6886 Assignment 2 — MobileNetV2 Quantization

This repository contains the implementation and experimental results for CS6886 Systems for Deep Learning Assignment 2. The work studies post-training quantization of a MobileNetV2 model trained for CIFAR-10 classification.

## Overview

The project has two main stages:

1. Train a FP32 MobileNetV2 baseline on CIFAR-10.
2. Apply configurable post-training linear quantization to weights and activations and compare different bit-widths.

The final preferred operating point is W6/A8 (6-bit weights, 8-bit activations), which achieved 90.61% test accuracy with a 0.30 percentage-point drop from the 90.91% FP32 baseline and an estimated 4.56x persistent-storage compression.

## Repository structure

```text
CS6886-MobileNetV2-Quantization/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── mobilenetv2_trained.ipynb
│   └── quantization.ipynb
├── figures/
│   ├── q1_training_test_accuracy.png
│   ├── q3_parallel_coordinates.png
│   └── q3_experiment_results.png
└── results/
    └── quantization_results.csv
```

## Environment

The experiments were run in Google Colab using a GPU runtime. The recorded notebook environment reports PyTorch 2.11.0+cu128. The quantization experiments used the CIFAR-10 dataset and a batch size of 128.

## Installation

```bash
pip install -r requirements.txt
```

For the notebooks, open them in Google Colab or a Jupyter environment with a CUDA-capable GPU when available.

## Running the notebooks

### 1. Train the baseline

Open:

```text
notebooks/mobilenetv2_trained.ipynb
```

Run the training cells to train MobileNetV2 for CIFAR-10. The recorded experiment used 20 epochs, Adam with learning rate 0.001, weight decay 1e-4, and cosine-annealing learning-rate scheduling.

### 2. Run quantization experiments

Open:

```text
notebooks/quantization.ipynb
```

The notebook:

- loads the trained FP32 checkpoint;
- implements uniform linear quantization;
- uses symmetric per-channel quantization for Conv2d/Linear weights;
- uses asymmetric per-tensor activation quantization;
- calibrates activation ranges using 100 batches from a deterministic CIFAR-10 calibration loader;
- evaluates multiple W/A bit-width combinations;
- estimates storage including quantized weights and quantization metadata.

The experiment configurations are W8/A8, W6/A8, W4/A8, W8/A6, W8/A4, W6/A6, and W4/A4.

## Reproducibility

The report numbers correspond to the recorded completed experiment. The training notebook did not expose a single seed-setting cell in the recorded workflow, so an exact global training seed is not claimed here. Quantization calibration uses a deterministic, non-shuffled calibration loader.

## Results

The complete quantization results are available in:

```text
results/quantization_results.csv
```

The figures folder contains the baseline accuracy curve, the experiment-results screenshot, and the W&B Parallel Coordinates visualization used in the report.

## Notes on model size

The reported compressed sizes are logical storage estimates assuming bit-packed quantized weights and FP32 scale/metadata where required. The current notebook evaluates quantization using fake quantization (quantize then dequantize), so the logical compressed size is not the same as the byte size of the fake-quantized PyTorch checkpoint.
