# Scripts Directory: Pipeline Orchestration & Experimental Framework

This directory contains the execution layer for the Math4AI Capstone project. It is structured to ensure a strict separation between **Data Engineering**, **Mathematical Validation**, and **Experimental Execution**, facilitating a modular and reproducible research workflow.

---

## 1. Core Components & Logic

### A. Data Engineering & Preprocessing
The foundation of the project relies on deterministic data generation to ensure results are comparable across different runs.

* **`generate_synthetic.py`**: 
    * **Objective**: Constructs 2D datasets with specific topological properties: **Linear Gaussian** and **Non-linear Moons**.
    * **Mathematics**: Utilizes Multivariate Normal distributions $\mathcal{N}(\mu, \Sigma)$ and non-convex manifolds to test the decision boundary limits of the Neural Network.
* **`make_digits_split.py`**:
    * **Objective**: Prepares the UCI Optical Recognition of Handwritten Digits dataset.
    * **Normalization**: Implements feature scaling to the $[0, 1]$ range. This is critical for preventing gradient vanishing in the `tanh` activation layers.
    * **Stratification**: Employs stratified sampling to maintain class proportions across Train/Val/Test splits, mitigating label shift issues.
* **`inspect_data.ipynb`**:
    * **Exploratory Data Analysis**: A Jupyter notebook used to inspect the shapes, class distributions, and value ranges of all data files before model implementation. Ensuring data integrity at the start of the project.

### B. Validation & Testing
* **`run_sanity_checks.py`**:
    * **Purpose**: Before running large experiments, this script verifies the mathematical correctness of the backpropagation.
    * **Mechanism**: It compares analytical gradients against numerical approximations to ensure the chain rule is correctly implemented.

### C. Experimental Execution
* **`run_experiments.py`**:
    * **Purpose**: The primary entry point for testing the three core tasks (Gaussian, Moons, and Digits).
    * **Output**: Automates the training-validation-testing cycle and logs accuracy/loss metrics.

* **`run_ablations.py`**:
    * **Optimization Analysis**: Compares different optimizers (e.g., SGD vs. Adam) and architectural capacities (hidden unit counts).
    * **Statistical Rigor**: Runs the models across **5 different random seeds** to compute mean performance and standard deviation.

* **`run_track_b.py`**:
    * **Objective**: Performs prediction confidence and reliability analysis.
    * **Advanced Track**: Implements 5-bin calibration tables and reliability diagrams to evaluate if the model "knows what it knows."

---

## 2. Mathematical Foundations

### Data Splitting Constraint
For every dataset, we enforce the following distribution invariant to ensure fairness:
$$P(y=k \mid S_{train}) \approx P(y=k \mid S_{val}) \approx P(y=k \mid S_{test})$$

### Forward Pass Logic
The scripts facilitate the training of a two-layer mapping function:
1. **Hidden Layer**: $Z^{(1)} = XW^{(1)} + b^{(1)} \rightarrow A^{(1)} = \tanh(Z^{(1)})$
2. **Output Layer**: $Z^{(2)} = A^{(1)}W^{(2)} + b^{(2)} \rightarrow \hat{y} = \text{Softmax}(Z^{(2)})$

---

## 3. Reproducibility Protocol

To ensure all results are verifiable, the following protocols are strictly enforced:
1. **Global Seeding**: A fixed `SEED = 7` is used to anchor the `numpy.random.Generator`.
2. **Path Resolution**: Scripts use `pathlib` to ensure cross-platform compatibility (Windows/Linux/macOS).
3. **Index Decoupling**: For the Digits dataset, split indices are saved as standalone binaries (`.npz`) to ensure identical sample usage across multiple sessions.

---

## 4. Execution Guide (Terminal Instructions)

Since this project utilizes a local virtual environment in PyCharm, use the following commands from within the `scripts/` directory to bypass Windows PATH issues:

### Step 1: Initialize Datasets
```powershell
..\..\.venv\Scripts\python.exe generate_synthetic.py
..\..\.venv\Scripts\python.exe make_digits_split.py
```
### Step 2: Run Mathematical Sanity Checks
```powershell
..\..\.venv\Scripts\python.exe run_sanity_checks.py
```
### Step 3: Run Core Experiments
```powershell
..\..\.venv\Scripts\python.exe run_experiments.py
```
### Step 4: Run Statistical Ablation Studies
```powershell
..\..\.venv\Scripts\python.exe run_ablations.py
```

### Step 5: Run Advanced Track B Analysis
```powershell
..\..\.venv\Scripts\python.exe run_track_b.py
```
