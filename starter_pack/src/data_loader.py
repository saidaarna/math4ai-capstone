import numpy as np

def load_synthetic(file_path):
    """
    Loads synthetic datasets (moons, linear_gaussian).
    These files come pre-split into train, val, and test sets.
    """
    data = np.load(file_path)
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def load_digits(data_path, split_path):
    """
    Specific loader for the MNIST digits data.
    Returns split digits arrays using fixed indices.
    """
    digits = np.load(data_path)
    splits = np.load(split_path)

    X = digits['X']
    y = digits['y']

    # Extracting indices from the fixed split file
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    return (X[train_idx], y[train_idx]), \
        (X[val_idx], y[val_idx]), \
        (X[test_idx], y[test_idx])










