import numpy as np

def train_softmax(model, optimizer, X_tr, y_tr, X_val, y_val,
                  epochs=200, batch_size=64):
    """
    Trains a Softmax Regression model using mini-batch gradient descent.
    Includes early-stopping logic by tracking the best validation loss checkpoint.
    """
    n = len(y_tr)
    
    # Dictionary to track performance metrics across epochs for later plotting
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    # Initialize variables to keep track of the best performing model state
    best_val_loss = np.inf
    best_W, best_b = model.W.copy(), model.b.copy()

    # Main training loop
    for epoch in range(epochs):
        
        # 1. Shuffle the training data at the start of each epoch.
        # This prevents the model from learning the sequence of the data 
        # and ensures gradients are noisy but randomly distributed.
        idx = np.random.permutation(n)
        
        # 2. Iterate through the shuffled data in small chunks (mini-batches)
        for i in range(0, n, batch_size):
            # Extract the current mini-batch
            Xb = X_tr[idx[i:i+batch_size]]
            yb = y_tr[idx[i:i+batch_size]]
            
            # Forward pass & Backpropagation on the mini-batch to get gradients
            dW, db = model.gradients(Xb, yb)
            
            # Update the model's weights and biases using the chosen optimizer (e.g., SGD, Adam)
            model.W, model.b = optimizer.update([model.W, model.b], [dW, db])

        # 3. Epoch Evaluation
        # Calculate loss and accuracy on the full training and validation sets
        tl = model.loss(X_tr, y_tr)
        vl = model.loss(X_val, y_val)
        va = model.accuracy(X_val, y_val)
        
        # Record the metrics
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)

        # 4. Model Selection / Checkpointing
        # If the model performs better on the validation set than ever before, save its state.
        # This ensures we don't return an overfitted model at the end of training.
        if vl < best_val_loss:
            best_val_loss = vl
            # IMPORTANT: Use .copy() so we save the specific values, 
            # not a reference to the array that is continually changing.
            best_W, best_b = model.W.copy(), model.b.copy()

    # 5. Restore Best Checkpoint
    # After all epochs are completed, load the weights that achieved the lowest validation loss
    model.W, model.b = best_W, best_b
    
    return history


def train_nn(model, optimizer, X_tr, y_tr, X_val, y_val,
             epochs=200, batch_size=64):
    """
    Trains a OneHiddenLayerNN using mini-batch gradient descent.
    Identical logic to train_softmax(), but handles 4 parameters:
    W1, b1 (hidden layer) and W2, b2 (output layer).
    """
    n = len(y_tr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # Save the best model checkpoint (all 4 weight arrays)
    best_val_loss = np.inf
    best_W1 = model.W1.copy()
    best_b1 = model.b1.copy()
    best_W2 = model.W2.copy()
    best_b2 = model.b2.copy()

    for epoch in range(epochs):

        # 1. Shuffle training data each epoch
        idx = np.random.permutation(n)

        # 2. Mini-batch updates
        for i in range(0, n, batch_size):
            Xb = X_tr[idx[i:i+batch_size]]
            yb = y_tr[idx[i:i+batch_size]]

            # Backprop returns gradients for all 4 parameter arrays
            dW1, db1, dW2, db2 = model.gradients(Xb, yb)

            # Pack params and grads into lists, update, then unpack
            params = [model.W1, model.b1, model.W2, model.b2]
            grads  = [dW1,      db1,      dW2,      db2]
            model.W1, model.b1, model.W2, model.b2 = optimizer.update(params, grads)

        # 3. Epoch evaluation
        tl = model.loss(X_tr, y_tr)
        vl = model.loss(X_val, y_val)
        va = model.accuracy(X_val, y_val)

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)

        # 4. Checkpoint: save best validation loss state
        if vl < best_val_loss:
            best_val_loss = vl
            best_W1 = model.W1.copy()
            best_b1 = model.b1.copy()
            best_W2 = model.W2.copy()
            best_b2 = model.b2.copy()

    # 5. Restore best checkpoint
    model.W1, model.b1, model.W2, model.b2 = best_W1, best_b1, best_W2, best_b2

    return history
