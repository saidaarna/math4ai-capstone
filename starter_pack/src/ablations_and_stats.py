import numpy as np
import os
# Importing required project modules
from data_loader import load_synthetic, load_digits
from neural_network import OneHiddenLayerNN
from optimizers import SGD, Momentum, Adam
from train import train_nn
from evaluate import evaluate
from plotting import plot_decision_boundary, plot_training_curves

# Ensure the output directory for plots exists
if not os.path.exists("figures"):
    os.makedirs("figures")

# =========================================================
# TASK 7.3: ABLATION STUDIES (Model Capacity & Optimizers)
# =========================================================
def run_ablation_studies():
    print(">>> Starting Task 7.3: Ablation Studies...")

    # Part A: Capacity Study on Moons Dataset
    # Using relative paths to navigate from src/ to data/
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_synthetic("../data/moons.npz")

    for h in [2, 8, 32]:
        print(f"Experiment: Hidden Units = {h}")
        model = OneHiddenLayerNN(d=2, hidden=h, k=2, lam=1e-4, seed=42)
        train_nn(model, SGD(lr=0.1), X_tr, y_tr, X_val, y_val)
        # Visualizing the decision boundaries for different capacities
        plot_decision_boundary(model, X_te, y_te, f"Capacity h={h}", f"figures/ablation_h{h}.png")

    # Part B: Optimizer Comparison on Digits Dataset
    print("\nExperiment: Comparing Optimizers on Digits")
    #Splitting loaded digits data into X and y for each subset
    train_data, val_data, test_data = load_digits(
        data_path="../data/digits_data.npz",
        split_path="../data/digits_split_indices.npz"
    )
    X_tr, y_tr = train_data
    X_val, y_val = val_data
    X_te, y_te = test_data

    optimizers = [
        ("SGD", SGD(lr=0.01)),
        ("Momentum", Momentum(lr=0.01, momentum=0.9)),
        ("Adam", Adam(lr=0.001))
    ]

    histories, names = [], []
    for name, opt in optimizers:
        print(f"Training model with {name}...")
        model = OneHiddenLayerNN(d=64, hidden=32, k=10, lam=1e-4, seed=42)
        history = train_nn(model, opt, X_tr, y_tr, X_val, y_val)
        histories.append(history)
        names.append(name)

        acc, _ = evaluate(model, X_te, y_te)
        print(f"-> {name} Test Accuracy: {acc:.4f}")

    # Save the training curves for all optimizers in one plot
    plot_training_curves(histories, names, "figures/optimizer_comparison.png")


# =========================================================
# TASK 7.4: REPEATED SEEDS (Statistical Analysis)
# =========================================================
def run_statistical_analysis():
    print("\n>>> Starting Task 7.4: Statistical Analysis (5 Seeds)...")

    # Reloading Digits dataset for consistency
    train_data, val_data, test_data = load_digits(
        data_path="../data/digits_data.npz",
        split_path="../data/digits_split_indices.npz"
    )
    X_tr, y_tr = train_data
    X_val, y_val = val_data
    X_te, y_te = test_data

    seeds = [0, 1, 2, 3, 4]
    accuracies = []

    for s in seeds:
        print(f"Running experiment with Seed {s}...")
        model = OneHiddenLayerNN(d=64, hidden=32, k=10, lam=1e-4, seed=s)
        # Using Adam as the primary optimizer for stability analysis
        train_nn(model, Adam(lr=0.001), X_tr, y_tr, X_val, y_val)
        acc, _ = evaluate(model, X_te, y_te)
        accuracies.append(acc)

    # Calculating the 95% Confidence Interval (CI)
    # Critical value (t*) for n=5 (df=4) and 95% CI is 2.776
    mean_acc = np.mean(accuracies)
    std_dev = np.std(accuracies, ddof=1)
    conf_interval = 2.776 * (std_dev / np.sqrt(len(seeds)))

    print(f"\nFinal Statistical Results (Digits Dataset):")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"95% Confidence Interval: ± {conf_interval:.4f}")
    print(f"Final Report Format: {mean_acc:.4f} ± {conf_interval:.4f}")


if __name__ == "__main__":
    run_ablation_studies()
    run_statistical_analysis()