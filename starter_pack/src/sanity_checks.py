import numpy as np
from softmax_regression import SoftmaxRegression
from neural_network import OneHiddenLayerNN

def run_sanity_checks():
    #  Check 1: Probability sum
    sr = SoftmaxRegression(d=64, k=10, lam=1e-4, seed=0)
    X_tiny = np.random.randn(10, 64)
    P = sr.forward(X_tiny)
    assert np.allclose(P.sum(axis=1), 1.0), "Probabilities don't sum to 1!"
    print(" Check 1: Probabilities sum to 1.")
    
    #  Check 2: Loss decreases on tiny subset
    from optimizers import SGD
    X5, y5 = X_tiny[:5], np.array([0,1,2,3,4])
    sr2 = SoftmaxRegression(d=64, k=10, lam=1e-4, seed=0)
    opt = SGD(lr=0.1)
    loss_before = sr2.loss(X5, y5)
    for _ in range(200):
        dW, db = sr2.gradients(X5, y5)
        sr2.W, sr2.b = opt.update([sr2.W, sr2.b], [dW, db])
    loss_after = sr2.loss(X5, y5)
    assert loss_after < loss_before, "Loss did not decrease!"
    print(f" Check 2: Loss decreased {loss_before:.3f} → {loss_after:.4f}")
    
    #  Check 3: Gradient numerical check (finite differences)
    sr3 = SoftmaxRegression(d=4, k=3, lam=1e-4, seed=0)
    X3, y3 = np.random.randn(8, 4), np.array([0,1,2,0,1,2,0,1])
    dW_analytic, _ = sr3.gradients(X3, y3)
    i, j = 0, 0 # check one entry
    eps = 1e-5
    sr3.W[i,j] += eps; L_plus = sr3.loss(X3, y3)
    sr3.W[i,j] -= 2*eps; L_minus = sr3.loss(X3, y3)
    sr3.W[i,j] += eps
    dW_numerical = (L_plus - L_minus) / (2*eps)
    rel_err = abs(dW_analytic[i,j] - dW_numerical) / (abs(dW_numerical) + 1e-8)
    assert rel_err < 1e-4, f"Gradient check failed: rel_err={rel_err:.2e}"
    print(f" Check 3: Gradient check passed (rel_err={rel_err:.2e})")
    
    #  Check 4: NaN/Inf check
    assert np.isfinite(sr3.loss(X3, y3)), "Loss is NaN or Inf!"
    print(" Check 4: No NaN/Inf in loss.")