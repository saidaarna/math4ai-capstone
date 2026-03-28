import numpy as np

class OneHiddenLayerNN:
    """
    A simple 1-hidden-layer neural network for classification.
    Uses Tanh activation for the hidden layer and Softmax for the output.
    """
    def __init__(self, d, hidden, k, lam=1e-4, seed=0):
        # Initialize a random number generator for reproducible results
        rng = np.random.default_rng(seed)
        
        # --- Layer 1 (Input to Hidden) ---
        # Weights (W1): Scaled by 0.1 to keep initial values small. Shape: (hidden_units, input_features)
        self.W1 = rng.standard_normal((hidden, d)) * 0.1
        # Biases (b1): Initialized to zero. Shape: (hidden_units,)
        self.b1 = np.zeros(hidden)
        
        # --- Layer 2 (Hidden to Output) ---
        # Weights (W2): Shape: (num_classes, hidden_units)
        self.W2 = rng.standard_normal((k, hidden)) * 0.1
        # Biases (b2): Initialized to zero. Shape: (num_classes,)
        self.b2 = np.zeros(k)
        
        # Regularization strength (lambda) to prevent overfitting
        self.lam = lam

    def forward(self, X):
        """
        Performs the forward pass through the network.
        X shape: (Number of samples, Input features).
        """
        self.X = X
        
        # --- Layer 1 ---
        # Linear transformation: Z1 = X * W1^T + b1
        self.Z1 = X @ self.W1.T + self.b1
        # Non-linear activation: Apply Hyperbolic Tangent (Tanh) to get hidden state H
        self.H  = np.tanh(self.Z1)
        
        # --- Layer 2 (Output Layer) ---
        # Linear transformation to get raw class scores (logits): S = H * W2^T + b2
        S  = self.H @ self.W2.T + self.b2
        
        # --- Softmax Activation ---
        # Numerical stability trick: subtract the max score of each row to prevent exponent overflow
        S -= S.max(axis=1, keepdims=True)
        E  = np.exp(S)
        # Calculate probabilities: P = exp(S) / sum(exp(S))
        self.P = E / E.sum(axis=1, keepdims=True)
        
        return self.P

    def loss(self, X, y):
        """
        Computes the Cross-Entropy loss with L2 Weight Regularization.
        """
        # Get network predictions (probabilities)
        P = self.forward(X)
        n = len(y)
        
        # Calculate Cross-Entropy Loss
        # Extract the predicted probability for the correct class using advanced indexing: P[np.arange(n), y]
        # Avoid log(0) by adding a tiny epsilon (1e-15)
        ce = -np.log(P[np.arange(n), y] + 1e-15).mean()
        
        # Calculate L2 Regularization penalty: (lambda / 2) * (sum(W1^2) + sum(W2^2))
        reg = 0.5 * self.lam * (np.sum(self.W1**2) + np.sum(self.W2**2))
        
        # Total loss is Data Loss + Regularization Loss
        return ce + reg

    def gradients(self, X, y):
        """
        Performs backpropagation to compute gradients for updating weights and biases.
        """
        n = len(y)
        
        # Run forward pass to get current probabilities
        P = self.forward(X)
        
        # Create a one-hot encoded matrix for the true labels Y
        Y = np.zeros_like(P)
        Y[np.arange(n), y] = 1.0
        
        # --- Backpropagation for Output Layer (Layer 2) ---
        # Gradient of categorical cross-entropy + softmax is simply (Probs - True_Labels).
        # We divide by 'n' to calculate the average gradient over the entire input batch.
        dS  = (P - Y) / n
        
        # Gradient of W2 = dS^T * H + (lambda * W2) for regularization
        dW2 = dS.T @ self.H  + self.lam * self.W2
        # Gradient of b2 = sum of dS across the batch
        db2 = dS.sum(axis=0)
        
        # --- Backpropagation for Hidden Layer (Layer 1) ---
        # Propagate gradients back through Layer 2 weights to the hidden state H
        dH  = dS @ self.W2
        
        # Gradient through the Tanh activation function: derivative of tanh(x) is (1 - tanh(x)^2)
        dZ1 = dH * (1 - self.H**2)
        
        # Gradient of W1 = dZ1^T * X + (lambda * W1) for regularization
        dW1 = dZ1.T @ X      + self.lam * self.W1
        # Gradient of b1 = sum of dZ1 across the batch 
        db1 = dZ1.sum(axis=0)
        
        return dW1, db1, dW2, db2

    def accuracy(self, X, y):
        """
        Calculates the classification accuracy.
        """
        # The predicted class is the index with the highest probability (argmax)
        preds = self.forward(X).argmax(axis=1)
        # Return the fraction of predictions that correctly match the true labels y
        return (preds == y).mean()
