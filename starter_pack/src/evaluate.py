import numpy as np

def evaluate(model, X, y):
    """
    Computes model performance on a given dataset.
    Returns: accuracy (float) and cross-entropy loss (float).
    """
    # 1. Calculate Cross-Entropy Loss using the model's internal method
    # This reflects how 'far' the predictions are from the true labels
    total_loss = model.loss(X, y)

    # 2. Calculate Accuracy
    # First, get the raw scores (logits) from the forward pass
    scores = model.forward(X)

    # Get the index of the highest score for each sample (the predicted class)
    predictions = np.argmax(scores, axis=1)

    # Compare predictions to actual labels (y) and calculate the mean
    accuracy = np.mean(predictions == y)

    return accuracy, total_loss