import numpy as np

def evaluate(model, X, y):
    """
    Computes model performance on a given dataset.
    Returns: accuracy (float) and cross-entropy loss (float).
    """
    # 1. Forward pass to get the full Probability matrix P
    # Unnormalized logits are transformed via softmax inside model.forward
    P = model.forward(X)
    n = len(y)

    # 2. Calculate purely unregularized Mean Cross-Entropy Loss
    # The assignment specifies: "average negative log-probability assigned to the true class"
    # We DO NOT use model.loss(X,y) here because that includes the L2 regularization penalty!
    cross_entropy = -np.log(P[np.arange(n), y] + 1e-15).mean()

    # 3. Calculate Accuracy
    # Get the index of the highest probability for each sample (the predicted class)
    predictions = np.argmax(P, axis=1)

    # Compare predictions to actual labels (y) and calculate the mean
    accuracy = np.mean(predictions == y)

    return accuracy, cross_entropy