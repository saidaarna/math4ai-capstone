import numpy as np
import os

def _resolve_path(path):
    if not os.path.isabs(path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, path)
    return path

def load_synthetic(file_path):
    """
    Loads synthetic datasets (moons, linear_gaussian).
    These files come pre-split into train, val, and test sets.
    """
    data = np.load(_resolve_path(file_path))
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def load_digits(data_path, split_path):
    """
    Specific loader for the MNIST digits data.
    Returns split digits arrays using fixed indices.
    """
    digits = np.load(_resolve_path(data_path))
    splits = np.load(_resolve_path(split_path))

    X = digits['X']
    y = digits['y']

    # Extracting indices from the fixed split file
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    return (X[train_idx], y[train_idx]), \
        (X[val_idx], y[val_idx]), \
        (X[test_idx], y[test_idx])










