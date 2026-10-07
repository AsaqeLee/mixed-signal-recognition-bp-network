# Mixed-Signal Recognition with Multiple BP Networks

MATLAB project for recognizing pairwise mixed modulated signals using multiple backpropagation (BP) neural networks specialized by SNR range.

## Overview

The system generates or loads mixed signals from common digital modulations, extracts features, trains separate BP networks for low / mid / high SNR regimes, and evaluates recognition performance. Using SNR-specialized networks aims to improve accuracy relative to a single network trained across the full SNR range.

## Scope

### Modulations (pairwise mixtures)

- 2ASK
- BPSK
- QPSK
- 16QAM

### SNR coverage

Approximately −5 dB to 20 dB in 5 dB steps (as configured in the generation scripts).

### Network specialization

| Network | SNR range |
|---------|-----------|
| Low | SNR < 5 dB |
| Mid | 5 dB ≤ SNR < 15 dB |
| High | SNR ≥ 15 dB |

### Typical network topology (per model)

- Input: feature dimension
- Hidden layers: 64 (ReLU) → 48 (ReLU) → 32 (Tanh) → 16
- Output: 8 neurons (Sigmoid), corresponding to pairwise mixture classes

Exact layer sizes and activations follow the training scripts; adjust them there if you change the feature set.

## Requirements

- MATLAB with Neural Network Toolbox (or Deep Learning Toolbox equivalents used by the scripts)
- Signal Processing / Communications toolboxes as needed for signal generation

## Getting started

From the repository root in MATLAB:

```matlab
run('main.m')
```

`main.m` orchestrates:

1. Dataset generation
2. Feature extraction
3. Training of the SNR-specialized BP networks
4. Evaluation
5. Result visualization

You may also call the supporting scripts individually after configuring paths and parameters.

## Project layout

```text
.
├── main.m
├── generate_dataset.m
├── generate_modulated_signal.m
├── extract_features.m
├── split_by_snr.m
├── train_multiple_networks.m
├── test_multiple_networks.m
├── data/                 # datasets, models, results
└── 项目解释.md           # Chinese project notes
```

## Usage notes

- For best results, estimate the SNR regime of an input and use the corresponding network; the pipeline can select a network automatically when SNR is known.
- Claims of improved accuracy vs a single network are design goals of this coursework-style project; re-run evaluation on your machine before citing numbers.

## Status / limitations

Academic / experimental MATLAB code. Not packaged as a production recognition service. Feature definitions and class counts are fixed by the scripts; extend them carefully if you add modulations.
