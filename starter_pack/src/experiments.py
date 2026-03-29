# Purpose: Compare Linear vs. Non-linear decision boundaries across three datasets.

from data_loader import load_synthetic, load_digits
from softmax_regression import SoftmaxRegression
from neural_network import OneHiddenLayerNN
from optimizers import SGD
from train import train_softmax, train_nn
from evaluate import evaluate
from plotting import plot_decision_boundary, plot_training_curves
import os

def _fig_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "..", "figures", filename)

def run_experiments():
    # =============================================================================
    # EXPERIMENT 1: Linear Gaussian
    # Context: Two classes generated from Gaussian distributions with equal covariance.
    # Theory: The optimal Bayesian decision boundary is linear: P(y|x) is a hyperplane.
    # =============================================================================
    
    # Load data: X_tr (N, d), y_tr (N,)
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_synthetic("../data/linear_gaussian.npz")
    
    # 1. Softmax Regression (Linear Baseline)
    # d=2 (input dims), k=2 (classes), lam=1e-4 (L2 regularization strength λ)
    sr_gauss = SoftmaxRegression(d=2, k=2, lam=1e-4, seed=0)
    opt_sr = SGD(lr=0.05)  # η = 0.05 (learning rate)
    hist_sr = train_softmax(sr_gauss, opt_sr, X_tr, y_tr, X_val, y_val)
    
    # 2. Neural Network (Non-linear Model)
    # hidden=32 (neurons in hidden layer m)
    nn_gauss = OneHiddenLayerNN(d=2, hidden=32, k=2, lam=1e-4, seed=0)
    opt_nn = SGD(lr=0.05)
    hist_nn = train_nn(nn_gauss, opt_nn, X_tr, y_tr, X_val, y_val)
    
    # Result Visualization
    # Since the boundary is linear, SR and NN should yield nearly identical test accuracy.
    plot_decision_boundary(sr_gauss, X_te, y_te, "SR — Gaussian", _fig_path("gaussian_sr.png"))
    plot_decision_boundary(nn_gauss, X_te, y_te, "NN — Gaussian", _fig_path("gaussian_nn.png"))
    
    print("Experiment 1 Complete: Expecting SR ≈ NN performance.")
    
    # =============================================================================
    # EXPERIMENT 2: Moons
    # Context: Interlocking semi-circles (non-linearly separable).
    # Theory: A linear model (SR) cannot solve this; a NN with tanh can approximate
    # the non-linear manifold.
    # =============================================================================
    
    X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m = load_synthetic("../data/moons.npz")
    
    # 1. Softmax Regression
    sr_moons = SoftmaxRegression(d=2, k=2, lam=1e-4, seed=0)
    train_softmax(sr_moons, SGD(lr=0.05), X_tr_m, y_tr_m, X_val_m, y_val_m)
    
    # 2. Neural Network
    nn_moons = OneHiddenLayerNN(d=2, hidden=32, k=2, lam=1e-4, seed=0)
    train_nn(nn_moons, SGD(lr=0.1), X_tr_m, y_tr_m, X_val_m, y_val_m,epochs=1000)
    
    # Result Visualization
    # The NN boundary should curve to wrap around the moons; the SR boundary will remain a straight line.
    plot_decision_boundary(sr_moons, X_te_m, y_te_m, "SR — Moons", _fig_path("moons_sr.png"))
    plot_decision_boundary(nn_moons, X_te_m, y_te_m, "NN — Moons", _fig_path("moons_nn.png"))
    
    print("Experiment 2 Complete: Expecting NN >> SR performance.")
    
    # =============================================================================
    # EXPERIMENT 3: Digits Benchmark (MNIST-like)
    # Context: 8x8 pixel images of digits 0-9. d=64, k=10.
    # Logic: L = -(1/n) Σ log(p_{y_i}) + (λ/2)(||W1||² + ||W2||²)
    # =============================================================================
    
    data_path = "../data/digits_data.npz"
    split_path = "../data/digits_split_indices.npz"
    
    (X_tr_d, y_tr_d), (X_val_d, y_val_d), (X_te_d, y_te_d) = load_digits(data_path, split_path)
    
    # Initialize Models
    sr_digit = SoftmaxRegression(d=64, k=10, lam=1e-4, seed=0)
    nn_digit = OneHiddenLayerNN(d=64, hidden=32, k=10, lam=1e-4, seed=0)
    
    # Training with history capture for dynamics plotting
    hist_sr_d = train_softmax(sr_digit, SGD(lr=0.05), X_tr_d, y_tr_d, X_val_d, y_val_d)
    hist_nn_d = train_nn(nn_digit, SGD(lr=0.05), X_tr_d, y_tr_d, X_val_d, y_val_d)
    
    # Quantitative Evaluation
    # Metrics: Accuracy (fraction correct) and Cross-Entropy (log-loss)
    for name, model in [('SR', sr_digit), ('NN', nn_digit)]:
        # Evaluate across all three splits (Training, Validation, Test)
        tr_acc, tr_ce = evaluate(model, X_tr_d, y_tr_d)
        va_acc, va_ce = evaluate(model, X_val_d, y_val_d)
        te_acc, te_ce = evaluate(model, X_te_d, y_te_d)
    
        print(f"\nModel: {name}")
        print(f"  Train: Acc={tr_acc:.3f}, CE={tr_ce:.3f}")
        print(f"  Val:   Acc={va_acc:.3f}, CE={va_ce:.3f}")
        print(f"  Test:  Acc={te_acc:.3f}, CE={te_ce:.3f}")
    
    # Training Dynamics Visualization
    # Plots Loss vs Epoch and Accuracy vs Epoch for both models on one canvas
    plot_training_curves([hist_sr_d, hist_nn_d], ['SR', 'NN'], _fig_path("training_dynamics.png"))
    
    print("\nExperiment 3 Complete: Dynamics saved to figures/training_dynamics.png")