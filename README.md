# CS6886 Assignment 2 - MobileNetV2 Quantization

This repository contains the implementation and experimental results for CS6886 Systems for Deep Learning Assignment 2. The work studies post-training quantization of a MobileNetV2 model trained for CIFAR-10 classification.

## Overview

The project has two stages:

1. Train a FP32 MobileNetV2 baseline on CIFAR-10.
2. Apply configurable post-training uniform linear quantization to weights and activations and compare multiple bit-widths.

The selected operating point is W6/A8 (6-bit weights, 8-bit activations). It achieved 90.61% test accuracy versus 90.91% for the recorded FP32 baseline, a 0.30 percentage-point drop, with an estimated 4.56x persistent-storage compression. These are the currently recorded results; after a fresh seeded baseline run, the report and CSV should be updated if the measured values change.

## Repository structure

```text
CS6886-MobileNetV2-Quantization/
├── README.md
├── requirements.txt
├── .gitattributes
├── checkpoints/
│   └── mobilenetv2_cifar10_fp32.pth
├── notebooks/
│   ├── mobilenetv2_trained.ipynb
│   └── quantization.ipynb
├── figures/
│   ├── basline_accuracy.png
│   ├── quantization_results_table.png
│   └── wandb_parallel_coordinates.png
├── results/
│   └── quantization_results.csv
├── scripts/
│   ├── evaluate.py
│   └── run_experiments.py
└── src/
    ├── data.py
    ├── evaluation.py
    ├── model.py
    ├── quantization.py
    └── storage.py
```

## Environment

The recorded experiments were run in Google Colab with a GPU runtime. The notebook environment reported PyTorch 2.11.0+cu128. The experiments used CIFAR-10, input size 224x224, and batch size 128.

## Installation

```bash
pip install -r requirements.txt
```

For GPU execution, use a CUDA-capable PyTorch installation. The code also runs on CPU, although the full sweep is slower.

## Checkpoint

The trained FP32 checkpoint is stored at:

```text
checkpoints/mobilenetv2_cifar10_fp32.pth
```

The checkpoint is tracked with Git LFS because it is a binary file of about 8.8 MB. After cloning the repository, make sure Git LFS is installed so the real checkpoint is pulled instead of only its pointer file:

```bash
git lfs install
git lfs pull
```

## Reproducing the experiments

### Baseline training

Open `notebooks/mobilenetv2_trained.ipynb` and run the cells in order. The notebook now fixes `SEED = 42`, seeds Python/NumPy/PyTorch, uses deterministic cuDNN settings, and seeds DataLoader workers. The training configuration is 20 epochs, Adam with learning rate 0.001, weight decay 1e-4, and cosine-annealing learning-rate scheduling.

Exact bit-for-bit reproducibility is not guaranteed across different hardware or software stacks, but the seed and deterministic settings make the experiment reproducible within the same environment as closely as PyTorch permits.

### Quantization sweep

Open `notebooks/quantization.ipynb` and run the cells in order. The notebook loads the FP32 checkpoint, calibrates activation ranges using 100 deterministic training batches, evaluates W8/A8, W6/A8, W4/A8, W8/A6, W8/A4, W6/A6, and W4/A4, and computes logical storage estimates.

The reusable command-line sweep is:

```bash
python scripts/run_experiments.py
```

For one configuration:

```bash
python scripts/evaluate.py --weight-bits 6 --activation-bits 8
```

By default the scripts use `checkpoints/mobilenetv2_cifar10_fp32.pth` and download CIFAR-10 into `./data` when needed.

## Quantization method

Weights use symmetric per-channel uniform quantization for Conv2d and Linear layers. Activations use asymmetric per-tensor uniform quantization. Activation ranges are collected during calibration and then frozen.

The implementation uses fake quantization for evaluation: values are quantized to the target integer levels and immediately dequantized before the normal PyTorch convolution or linear operation. Therefore the reported accuracy measures the effect of quantization noise, while the storage numbers are logical packed-storage estimates. This repository does not claim a fully integer-only execution backend.

Biases and BatchNorm tensors remain in FP32.

## Reproducibility note

The calibration/evaluation data loader is deterministic and non-shuffled. The baseline training notebook now uses a fixed global seed (`SEED = 42`), deterministic PyTorch settings, and seeded DataLoader workers. The historical 90.91% FP32 result was produced before this seed was added, so a fresh seeded run may produce slightly different metrics. Any changed measured result should replace the corresponding report/CSV value rather than being silently treated as identical.

## Results

The full sweep is available in `results/quantization_results.csv`. The W6/A8 operating point is used in the report because it provides a small accuracy loss with substantially lower persistent representation than FP32.

W&B Parallel Coordinates visualization:

https://wandb.ai/tejaswi6874-iit-madras/CS6886-MobileNetV2-Quantization/runs/puxyh7i9

## Model-size interpretation

The reported compressed sizes include quantized Conv2d/Linear weights, FP32 per-channel weight scales, remaining FP32 parameters, FP32 buffers, and activation scale/zero-point metadata. They are logical storage estimates under the stated assumptions and should not be confused with the byte size of the fake-quantized PyTorch model object.
