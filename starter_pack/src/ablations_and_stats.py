import numpy as np
import os
# Importing required project modules
from data_loader import load_synthetic, load_digits
from neural_network import OneHiddenLayerNN
from softmax_regression import SoftmaxRegression
from optimizers import SGD, Momentum, Adam
from train import train_nn, train_softmax
from evaluate import evaluate
from plotting import plot_decision_boundary, plot_training_curves

def _fig_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "..", "figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    return os.path.join(fig_dir, filename)

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
        # Use 1000 epochs + lr=0.1 (same as main moons experiment) so each network
        # has enough budget to express its maximum representational capacity.
        # With fewer epochs all networks look linear because tanh hasn't had time to curve.
        train_nn(model, SGD(lr=0.1), X_tr, y_tr, X_val, y_val, epochs=1000)
        # Visualizing the decision boundaries for different capacities
        plot_decision_boundary(model, X_te, y_te, f"Capacity h={h}", _fig_path(f"ablation_h{h}.png"))

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
        ("SGD", SGD(lr=0.05)),
        ("Momentum", Momentum(lr=0.05, momentum=0.9)),
        ("Adam", Adam(lr=0.001))
    ]

    histories, names = [], []
    for name, opt in optimizers:
        print(f"Training model with {name}...")
        model = OneHiddenLayerNN(d=64, hidden=32, k=10, lam=1e-4, seed=42)
        history = train_nn(model, opt, X_tr, y_tr, X_val, y_val)
        histories.append(history)
        names.append(name)

        acc, ce = evaluate(model, X_te, y_te)
        print(f"-> {name} Test Accuracy: {acc:.4f} | Cross-Entropy: {ce:.4f}")

    # Save the training curves for all optimizers in one plot
    plot_training_curves(histories, names, _fig_path("optimizer_comparison.png"))


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
    
    sr_accs, sr_ces = [], []
    nn_accs, nn_ces = [], []

    for s in seeds:
        print(f"Running experiment with Seed {s}...")
        
        # Softmax Regression Benchmark Configuration
        sr_model = SoftmaxRegression(d=64, k=10, lam=1e-4, seed=s)
        train_softmax(sr_model, SGD(lr=0.05), X_tr, y_tr, X_val, y_val)
        acc, ce = evaluate(sr_model, X_te, y_te)
        sr_accs.append(acc)
        sr_ces.append(ce)
        
        # Neural Network Benchmark Configuration (Adam was historically fastest here)
        nn_model = OneHiddenLayerNN(d=64, hidden=32, k=10, lam=1e-4, seed=s)
        train_nn(nn_model, Adam(lr=0.001), X_tr, y_tr, X_val, y_val)
        acc, ce = evaluate(nn_model, X_te, y_te)
        nn_accs.append(acc)
        nn_ces.append(ce)

    # Printing 95% Confidence Interval (CI)
    # Critical value (t*) for n=5 (df=4) and 95% CI is 2.776
    print("\n--- Final Statistical Results (Digits Dataset) ---")
    
    for name, accs, ces in [("Softmax Regression", sr_accs, sr_ces), 
                            ("1-Hidden-Layer NN", nn_accs, nn_ces)]:
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        ci_acc = 2.776 * (std_acc / np.sqrt(5))
        
        mean_ce = np.mean(ces)
        std_ce = np.std(ces, ddof=1)
        ci_ce = 2.776 * (std_ce / np.sqrt(5))
        
        print(f"\nModel: {name}")
        print(f"Accuracy:      {mean_acc:.4f} ± {ci_acc:.4f}")
        print(f"Cross-Entropy: {mean_ce:.4f} ± {ci_ce:.4f}")


if __name__ == "__main__":
    run_ablation_studies()
    run_statistical_analysis()