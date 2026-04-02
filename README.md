# Math4AI Final Capstone — From Linear Scores to a Single Hidden Layer

**National AI Center · AI Academy**

> *When does a one-hidden-layer nonlinear classifier genuinely improve on a linear decision rule, and when is additional model complexity unnecessary?*

This repository contains our team's full implementation, experiments, and report for the Math4AI final capstone project. All models, training loops, and evaluation metrics are implemented **from scratch in NumPy only**. No PyTorch, TensorFlow, JAX, or scikit-learn model classes are used.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Team](#2-team)
3. [Environment Setup](#3-environment-setup)
4. [Repository Structure](#4-repository-structure)
5. [How to Reproduce Experiments](#5-how-to-reproduce-experiments)
6. [Models Implemented](#6-models-implemented)
7. [Datasets](#7-datasets)
8. [Experimental Protocol](#8-experimental-protocol)
9. [Key Results](#9-key-results)

---

## 1. Project Objective

We compare **Softmax Regression (SR)** — a linear probabilistic classifier — against a **one-hidden-layer neural network (NN)** with `tanh` activations and softmax output across three tasks:

| Task | Type | Result |
|---|---|---|
| Linear Gaussian | Synthetic, linearly separable | SR ≈ NN (linear boundary sufficient) |
| Moons | Synthetic, nonlinearly separable | SR fails, NN succeeds |
| Digits benchmark | Real, 64-dim, 10 classes | NN statistically significantly better |

We additionally complete **Track B** reliability analysis, showing that the NN is not only more accurate but also better calibrated.

---

## 2. Team

| Member | Role |
|---|---|
| Nazrin Aliyeva |  Track B reliability analysis, Sanity checking, Plotting, Reporting, Repository Management |
| Nigar Rustamova | Data Loading, Softmax regression implementation, Experiments, Reporting, Repository Management |
| Laman Mirzayeva |  Optimization, Evaluation, Ablations and Statistics , Reporting, Repository Management |
| Saida Arabova | Data Inspection, Neural network implementation, Model Training, Reporting, Repository Management |

---

## 3. Environment Setup

**Requirements:** Python 3.9+, NumPy, Matplotlib

```powershell
# Clone the repository
git clone https://github.com/saidaarna/math4ai-capstone.git
cd math4ai-capstone

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# Install dependencies
pip install numpy matplotlib scikit-learn
```

> **Allowed libraries:** `numpy`, `matplotlib`, `scikit-learn` (data utilities only, not model classes).

---

## 4. Repository Structure

```
math4ai_capstone/
│
├── deliverables/
│   └── math4ai_capstone_assignment.tex   # Official capstone handout
│
└── starter_pack/
    ├── README.md                         # Starter-pack overview and checklist pointer
    ├── CHECKLIST.md                      # Task-by-task completion checklist
    │
    ├── data/                             # Fixed datasets (do not modify)
    │   ├── digits_data.npz               # 64-dim digit features and labels
    │   ├── digits_split_indices.npz      # Fixed train/val/test split indices
    │   ├── linear_gaussian.npz           # Linearly separable Gaussian dataset
    │   └── moons.npz                     # Nonlinearly separable moons dataset
    │
    ├── src/                              # Core library (no execute-at-import code)
    │   ├── softmax_regression.py         # SoftmaxRegression class
    │   ├── neural_network.py             # OneHiddenLayerNN class (tanh + softmax)
    │   ├── optimizers.py                 # SGD, Momentum, Adam
    │   ├── train.py                      # train_softmax(), train_nn() with checkpointing
    │   ├── evaluate.py                   # Accuracy and unregularized cross-entropy
    │   ├── data_loader.py                # Dataset loading utilities
    │   ├── experiments.py                # Core experiment logic (Gaussian/Moons/Digits)
    │   ├── ablations_and_stats.py        # Capacity ablation, optimizer study, 5-seed stats
    │   ├── sanity_checks.py              # Gradient verification, probability checks
    │   └── track_b.py                    # Track B: confidence and reliability analysis
    │
    ├── scripts/                          # Entry-point runners (execute these)
    |   ├── inspect_data.ipynb            # Data inspection notebook → saves to figures/data_inspect_figures/
    │   ├── run_experiments.py            # Gaussian, Moons, Digits experiments
    │   ├── run_ablations.py              # Capacity ablation + optimizer study + 5-seed stats
    │   ├── run_sanity_checks.py          # Implementation verification checks
    │   ├── run_track_b.py                # Track B reliability analysis
    │   ├── make_digits_split.py          # Reproducible split generation
    │   └── generate_synthetic.py         # Synthetic dataset generation
    │
    ├── figures/                          # All generated plots (PNG)
    │   └── data_inspect_figures/         # Dataset exploration plots from 
    ├── results/                          # Numerical logs saved automatically on each run
    ├── report/
    │   ├── main.tex                      # Complete LaTeX report (Overleaf-ready)
    │   └── template.tex                  # Original optional template
    └── slides/                           # Presentation slides
```

---

## 5. How to Reproduce Experiments

Run all scripts from the **repository root** with the virtual environment active.

```powershell
# Step 0 — Inspect datasets and generate exploration figures
jupyter nbconvert --to notebook --execute starter_pack/scripts/inspect_data.ipynb
# (or open in VS Code / Jupyter and run all cells)

# Step 1 — Verify implementation correctness
python starter_pack/scripts/run_sanity_checks.py

# Step 2 — Run all core experiments (Gaussian, Moons, Digits)
python starter_pack/scripts/run_experiments.py

# Step 3 — Run ablations, optimizer study, and 5-seed statistics
python starter_pack/scripts/run_ablations.py

# Step 4 — Run Track B reliability analysis
python starter_pack/scripts/run_track_b.py
```

Each script **automatically saves** its full console output as a `.txt` log to `starter_pack/results/`, and saves all figures to `starter_pack/figures/`.

---

## 6. Models Implemented

### Softmax Regression (Baseline)
- Linear score function: `s(x) = Wx + b`
- Softmax output: `p_j = exp(s_j) / Σ exp(s_ℓ)`
- L2 regularization with λ = 1e-4
- Xavier uniform weight initialization

### One-Hidden-Layer Neural Network
- Forward pass: `h = tanh(W₁x + b₁)`, `s = W₂h + b₂`, `p = softmax(s)`
- Full vectorized backpropagation derived from the chain rule
- L2 regularization on both weight matrices
- Xavier uniform initialization per layer

### Optimizers
- **SGD**: `θ ← θ − η∇`
- **Momentum**: velocity accumulation with `μ = 0.9`
- **Adam**: adaptive per-parameter learning rates (β₁=0.9, β₂=0.999)

---

## 7. Datasets

| File | Description | Split |
|---|---|---|
| `digits_data.npz` | 1797 samples, 64 features (8×8 pixels), 10 classes | Fixed via `digits_split_indices.npz` |
| `digits_split_indices.npz` | Shared train/val/test index arrays for all teams | — |
| `linear_gaussian.npz` | Two Gaussian blobs, d=2, k=2 | Provided |
| `moons.npz` | Interlocking crescents, d=2, k=2 | Provided |

> **Important:** Digits features are pre-scaled to [0, 1]. No additional normalization is applied. The split indices in `digits_split_indices.npz` are fixed for all experiments.

---

## 8. Experimental Protocol

The digits benchmark follows a strict fixed contract (per assignment requirements):

| Setting | Value |
|---|---|
| Optimizer (SR default) | SGD, η = 0.05 |
| Optimizer (NN default) | SGD, η = 0.05 |
| Batch size | 64 |
| Epoch budget | 200 |
| L2 regularization λ | 1e-4 |
| Default hidden width | 32 |
| Model selection metric | Validation cross-entropy |
| Checkpoint rule | Best validation cross-entropy within 200 epochs |
| Repeated seeds | 5 seeds (0–4), 95% CI: x̄ ± 2.776 · s/√5 |

---

## 9. Key Results

### Digits Benchmark — Single Run (seed 0)

| Model | Test Accuracy | Test CE |
|---|---|---|
| Softmax Regression | 0.938 | 0.268 |
| 1-Hidden-Layer NN | 0.951 | 0.163 |

### Repeated-Seed Statistics (5 seeds, 95% CI)

| Model | Test Accuracy | Test CE |
|---|---|---|
| Softmax Regression | 0.9375 ± 0.0024 | 0.2683 ± 0.0027 |
| 1-Hidden-Layer NN | 0.9511 ± 0.0000 | 0.1671 ± 0.0077 |

Non-overlapping confidence intervals confirm the NN improvement is statistically reliable.

### Track B — Calibration (highest-confidence bin)

| Model | Predictions in [0.80,1.00] | Empirical Accuracy | Gap |
|---|---|---|---|
| SR | 273 / 368 (74%) | 1.000 | −0.064 |
| NN | 325 / 368 (88%) | 0.991 | **−0.016** |

### Additional Insights
- **Capacity limits don't always matter:** Our experiment showed that on a simple dataset (Moons) with optimized training, using a complex model (h=32) gave us no increase in validation precision (0.9875) compared to h=2. The h=2 model was sufficient to warp the space as needed.
- **Optimization limits are critical:** When training with default optimizer settings ($\eta=0.05$, 200 epochs) on the Moons task, the validation accuracy completely stalled at exactly 0.8875 across all hidden widths (m=2, 8, and 32), leaving the decision boundary near-linear despite massive capacity.
- **Overconfident when wrong:** Although the Neural Network has higher absolute accuracy than SR, it can occasionally be more overconfident when giving wrong answers.

---


## References

- Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*, Cambridge University Press, 2020
- Murphy — *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022
- Goodfellow, Bengio & Courville — *Deep Learning*, MIT Press, 2016
- Stanford CS229 Deep Learning Notes — https://cs229.stanford.edu/
- Stanford CS231n Backpropagation Notes — https://cs231n.github.io/optimization-2/

