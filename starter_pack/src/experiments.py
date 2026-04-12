# experiments.py
import os
from data_loader import load_synthetic, load_digits
from softmax_regression import SoftmaxRegression
from neural_network import OneHiddenLayerNN
from optimizers import SGD
from train import train_softmax, train_nn
from evaluate import evaluate
from plotting import plot_decision_boundary, plot_training_curves


# =============================================================================
# Helper: Figure path
# =============================================================================
def _fig_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "..", "figures", filename)


# =============================================================================
# Helper: Hyperparameter tuning for NN using validation set
# =============================================================================
def tune_nn_with_validation(X_tr, y_tr, X_val, y_val):
    print("\n🔍 Starting Hyperparameter Tuning (using validation set)...\n")

    best_val_acc = -1
    best_config = None
    best_model = None
    best_history = None

    learning_rates = [0.01, 0.05, 0.1]
    lambdas = [1e-4, 1e-5, 1e-6]
    epochs_list = [200, 500, 1000]

    for lr in learning_rates:
        for lam in lambdas:
            for epochs in epochs_list:
                print(f"Testing: lr={lr}, lambda={lam}, epochs={epochs}")

                # Re-initialize model for each combination
                model = OneHiddenLayerNN(d=2, hidden=32, k=2, lam=lam, seed=0)
                optimizer = SGD(lr=lr)

                history = train_nn(model, optimizer, X_tr, y_tr, X_val, y_val, epochs=epochs)
                val_acc = max(history['val_acc'])
                print(f"→ Best Val Acc: {val_acc:.4f}\n")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = (lr, lam, epochs)
                    best_model = model
                    best_history = history

    print("✅ BEST CONFIG FOUND:")
    print(f"   Learning Rate: {best_config[0]}")
    print(f"   Lambda: {best_config[1]}")
    print(f"   Epochs: {best_config[2]}")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f}\n")

    return best_model, best_history, best_config


# =============================================================================
# Main experiment runner
# =============================================================================
def run_experiments():
    # =============================================================================
    # Experiment 1: Linear Gaussian
    # =============================================================================
    print("\n===== Experiment 1: Linear Gaussian =====")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_synthetic("../data/linear_gaussian.npz")

    # Softmax Regression (SR)
    sr_gauss = SoftmaxRegression(d=2, k=2, lam=1e-4, seed=0)
    lr = 0.05
    hist_sr = train_softmax(sr_gauss, SGD(lr=lr), X_tr, y_tr, X_val, y_val)

    # Neural Network (NN)
    nn_gauss = OneHiddenLayerNN(d=2, hidden=32, k=2, lam=1e-4, seed=0)
    hist_nn = train_nn(nn_gauss, SGD(lr=lr), X_tr, y_tr, X_val, y_val)

    # Visualization
    plot_decision_boundary(sr_gauss, X_te, y_te, "SR — Gaussian", _fig_path("gaussian_sr.png"))
    plot_decision_boundary(nn_gauss, X_te, y_te, "NN — Gaussian", _fig_path("gaussian_nn.png"))
    plot_training_curves(
        [hist_sr, hist_nn],
        ['SR', 'NN'],
        _fig_path("gaussian_dynamics.png"),
        title="Training Dynamics: Linear Gaussian Task"
    )

    # Evaluation
    acc_sr, _ = evaluate(sr_gauss, X_te, y_te)
    acc_nn, _ = evaluate(nn_gauss, X_te, y_te)
    print("\n📊 Experiment 1 Evaluation (Gaussian):")
    print(f"Softmax Regression Test Accuracy: {acc_sr:.4f}")
    print(f"Neural Network Test Accuracy: {acc_nn:.4f}\n")

    # =============================================================================
    # Experiment 2: Moons
    # =============================================================================
    print("\n===== Experiment 2: Moons =====")
    X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m = load_synthetic("../data/moons.npz")

    # Softmax Regression (SR)
    sr_moons = SoftmaxRegression(d=2, k=2, lam=1e-4, seed=0)
    hist_sr_m = train_softmax(sr_moons, SGD(lr=0.05), X_tr_m, y_tr_m, X_val_m, y_val_m)

    # Neural Network (default)
    nn_moons_def = OneHiddenLayerNN(d=2, hidden=32, k=2, lam=1e-4, seed=0)
    hist_nn_def = train_nn(nn_moons_def, SGD(lr=0.05), X_tr_m, y_tr_m, X_val_m, y_val_m, epochs=200)

    # # Neural Network (tuned using validation)
    # nn_moons_opt, hist_nn_opt, best_config = tune_nn_with_validation(X_tr_m, y_tr_m, X_val_m, y_val_m)
    # print(
    #     f"Best hyperparameters found for Moons: lr={best_config[0]}, lambda={best_config[1]}, epochs={best_config[2]}")
    #
    # # Visualization
    # plot_decision_boundary(sr_moons, X_te_m, y_te_m, "SR — Moons (Linear)", _fig_path("moons_sr.png"))
    # plot_decision_boundary(nn_moons_def, X_te_m, y_te_m, "NN — Moons (Default/Failed)",
    #                        _fig_path("moons_nn_default.png"))
    # plot_decision_boundary(nn_moons_opt, X_te_m, y_te_m, "NN — Moons (Tuned)", _fig_path("moons_nn_tuned.png"))
    # plot_training_curves(
    #     [hist_sr_m, hist_nn_def, hist_nn_opt],
    #     ['SR Baseline', 'NN Default', 'NN Tuned'],
    #     _fig_path("moons_comparison_dynamics.png"),
    #     title="Moons Task: Validation-Based Tuning"
    # )
    #
    # # Evaluation
    # acc_sr_m, _ = evaluate(sr_moons, X_te_m, y_te_m)
    # acc_def, _ = evaluate(nn_moons_def, X_te_m, y_te_m)
    # acc_opt, _ = evaluate(nn_moons_opt, X_te_m, y_te_m)
    # print("\n📊 Experiment 2 Evaluation (Moons):")
    # print(f"Softmax Regression (Linear) Test Accuracy: {acc_sr_m:.4f}")
    # print(f"Neural Network (Default) Test Accuracy: {acc_def:.4f}")
    # print(f"Neural Network (Tuned) Test Accuracy: {acc_opt:.4f}\n")

    # --- Experiment 2: Neural Network (Manual Configuration) ---

    # 1. Initialize the model with the specified parameters
    # Parameters: d=2 (input), hidden=32, k=2 (output), lambda=1e-4
    nn_moons_opt = OneHiddenLayerNN(d=2, hidden=32, k=2, lam=1e-4, seed=0)

    # 2. Train the model using SGD with lr=0.05 for 1000 epochs
    hist_nn_opt = train_nn(
        nn_moons_opt,
        SGD(lr=0.05),
        X_tr_m, y_tr_m,
        X_val_m, y_val_m,
        epochs=800
    )

    print("Manual Hyperparameters: lr=0.05, lambda=1e-4, hidden=32, epochs=1000")

    # --- Visualization ---

    # Plot the Linear Baseline
    plot_decision_boundary(sr_moons, X_te_m, y_te_m, "SR — Moons (Linear)", _fig_path("moons_sr.png"))

    # Plot the Neural Network with your manual settings
    plot_decision_boundary(nn_moons_opt, X_te_m, y_te_m, "NN — Moons (Manual/Tuned)", _fig_path("moons_nn_tuned.png"))

    # Comparison of training curves (SR vs NN)
    plot_training_curves(
        [hist_sr_m, hist_nn_opt],
        ['SR Baseline', 'NN Manual'],
        _fig_path("moons_comparison_dynamics.png"),
        title="Moons Task: Manual NN Training"
    )

    # --- Evaluation ---

    acc_sr_m, _ = evaluate(sr_moons, X_te_m, y_te_m)
    acc_opt, _ = evaluate(nn_moons_opt, X_te_m, y_te_m)

    print("\n📊 Experiment 2 Evaluation (Moons):")
    print(f"Softmax Regression (Linear) Test Accuracy: {acc_sr_m:.4f}")
    print(f"Neural Network (Manual/Tuned) Test Accuracy: {acc_opt:.4f}\n")
    # =============================================================================
    # Experiment 3: Digits Benchmark
    # =============================================================================
    print("\n===== Experiment 3: Digits Benchmark =====")
    data_path = "../data/digits_data.npz"
    split_path = "../data/digits_split_indices.npz"
    (X_tr_d, y_tr_d), (X_val_d, y_val_d), (X_te_d, y_te_d) = load_digits(data_path, split_path)

    # Models
    sr_digit = SoftmaxRegression(d=64, k=10, lam=1e-4, seed=0)
    nn_digit = OneHiddenLayerNN(d=64, hidden=32, k=10, lam=1e-4, seed=0)

    # Training
    hist_sr_d = train_softmax(sr_digit, SGD(lr=0.05), X_tr_d, y_tr_d, X_val_d, y_val_d)
    hist_nn_d = train_nn(nn_digit, SGD(lr=0.05), X_tr_d, y_tr_d, X_val_d, y_val_d)

    # Evaluation
    for name, model in [('SR', sr_digit), ('NN', nn_digit)]:
        tr_acc, tr_ce = evaluate(model, X_tr_d, y_tr_d)
        va_acc, va_ce = evaluate(model, X_val_d, y_val_d)
        te_acc, te_ce = evaluate(model, X_te_d, y_te_d)
        print(f"\nModel: {name}")
        print(f"  Train: Acc={tr_acc:.3f}, CE={tr_ce:.3f}")
        print(f"  Val:   Acc={va_acc:.3f}, CE={va_ce:.3f}")
        print(f"  Test:  Acc={te_acc:.3f}, CE={te_ce:.3f}")

    # Plots
    plot_training_curves(
        [hist_sr_d, hist_nn_d],
        ['SR', 'NN'],
        _fig_path("digits_training_dynamics.png"),
        title="Training Dynamics: Digits Benchmark"
    )
    print("\n✅ All Experiments Complete.")