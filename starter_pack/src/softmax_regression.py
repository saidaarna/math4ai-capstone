import numpy as np

class SoftmaxRegression:
    def __init__(self, d, k, lam=1e-4, seed=0):
        """
        Initializes a Multiclass Linear Classifier (Baseline Model).
        d: Number of input features (e.g., 64 for digits).
        k: Number of target classes (e.g., 10 for digits).
        lam: L2 regularization coefficient (lambda) for weight decay.
        """
        # Reproducibility via seeded random number generation.
        rng = np.random.default_rng(seed)

        # Weights W initialized to shape (k, d).
        # Initialize weights using the Xavier/Glorot criterion (U[-limit, limit]).
        # This scales the initial weights according to the input (d) and output (k) dimensions,
        # preserving the variance of the signal to prevent vanishing gradients and
        # ensuring that the initial softmax probabilities remain non-extreme.
        limit = np.sqrt(6 / (d + k))
        self.W = rng.uniform(-limit, limit, (k, d))
        self.b = np.zeros(k)
        self.lam = lam

    def forward(self, X):
        """
        Computes the forward pass: maps input features to a probability distribution.
        Returns P: Probability matrix of shape (n, k).
        """
        # Section 3.8: Linear score function s(x) = XW^T + b.
        # This produces 'Logits'—the unnormalized log-probability scores.
        S = X @ self.W.T + self.b

        # Numerical Care: Subtracting the row-wise maximum prevents 'exploding' values in np.exp(S) without altering the softmax ratio.
        # exp(s_j - C) / sum(exp(s_l - C))
        S -= S.max(axis=1, keepdims=True)

        # The Softmax Map.
        # Normalizes exponents so that each row is non-negative and sums to 1.0.
        E = np.exp(S)
        P = E / E.sum(axis=1, keepdims=True)
        return P

    def loss(self, X, y):
        """
        Computes Total Loss: The sum of Negative Log-Likelihood and L2 Penalty.
        """
        P = self.forward(X)
        n = len(y)

        # Cross-Entropy Loss L = -log(Py).
        # Indexing [np.arange(n), y] to select the model's predicted probability for the ground-truth labels.
        # 1e-15 (epsilon) prevents log(0) errors.
        ce = -np.log(P[np.arange(n), y] + 1e-15).mean()

        # Regularization: Adds a penalty proportional to the squared Frobenius norm of W.
        # This constrains model complexity and prevents overfitting.
        reg = 0.5 * self.lam * np.sum(self.W ** 2)

        return ce + reg

    def gradients(self, X, y):
        """
        Computes analytical gradients via the Chain Rule.
        Used by the optimizer (SGD, Momentum, or Adam) to update the model parameters.
        """
        n = len(y)
        P = self.forward(X)

        # Transform integer labels into a One-Hot encoded matrix Y.
        Y = np.zeros_like(P)
        Y[np.arange(n), y] = 1.0

        # The gradient of the Softmax Cross-Entropy loss with respect to the scores (S) is simply (Predictions - Truth).
        dS = (P - Y) / n

        # Gradient of W: Includes the 'Weight Decay' term (lambda * W) derived from the L2 regularization penalty.
        dW = dS.T @ X + self.lam * self.W

        # Gradient of b: The sensitivity of the loss to the bias shifts.
        db = dS.sum(axis=0)

        return dW, db

    def accuracy(self, X, y):
        """
        Evaluates performance using the zero-one loss metric.
        """
        # Predicted class is the index with the maximal logit.
        preds = self.forward(X).argmax(axis=1)
        return (preds == y).mean()