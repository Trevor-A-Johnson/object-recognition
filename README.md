# CIFAR-10 Object Recognition

A machine learning project that trains and compares two CNN architectures on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) for image classification.

## Project Overview

CIFAR-10 consists of 60,000 32x32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). This project:

- Loads and preprocesses the dataset with data augmentation
- Trains two CNN models with different architectures (SimpleCNN and DeepCNN)
- Evaluates both models and compares performance using accuracy, 95% confidence intervals, normalized confusion matrices, and per-class accuracy

## Models

| Model | Architecture | Parameters |
|---|---|---|
| **SimpleCNN** | 2 conv layers, MaxPool, FC | ~1.07M |
| **DeepCNN** | 3 conv layers + BatchNorm + Dropout, FC | ~2.19M |

## Files

| File | Description |
|---|---|
| `cifar10_training.ipynb` | Main notebook — data loading, training, evaluation, plots |
| `cifar10_training.py` | Script version of the notebook — same functionality, runnable from the terminal |
| `extractData.py` | Utility to load raw CIFAR-10 batch files |
| `simple_cnn.pth` | Saved SimpleCNN weights |
| `deep_cnn.pth` | Saved DeepCNN weights |
| `training_curves.png` | Loss curves for both models |
| `confusion_matrices.png` | Normalized confusion matrices for both models |
| `confidence_intervals.png` | Bar chart of model accuracy with 95% confidence intervals |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

### Notebook (`cifar10_training.ipynb`)
Open in VS Code or JupyterLab and run cells top to bottom. To skip training and use saved weights, run sections 1–4 and 7–12 (skip sections 5–6).

### Script (`cifar10_training.py`)

Train from scratch (saves weights to `simple_cnn.pth` / `deep_cnn.pth`):
```bash
python3 cifar10_training.py --train
```

Evaluate using saved weights (skips training):
```bash
python3 cifar10_training.py
```

Both modes run evaluation, plot confidence intervals, confusion matrices, and per-class accuracy. Training curves are only produced when `--train` is passed.

## Dataset

Download the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html and extract it so that `cifar-10-batches-py/` is in the project root. If the folder is already present, set `download=False` in the data loading cell (default).

## Project Requirements

- Load and preprocess CIFAR-10 data with normalization and augmentation
- Train and evaluate at least two models
- Comprehensively compare model performance
- Written report (IEEE format) including: abstract, introduction, problem description, model descriptions, performance comparison with confidence intervals and confusion matrices, and pros/cons analysis
