#!/usr/bin/env python3
"""
Run Track B: Prediction Confidence and Reliability Analysis.
Usage: python scripts/run_track_b.py
"""
import sys
import os

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
sys.stdout = DualLogger(os.path.join(results_dir, os.path.basename(__file__).replace('.py', '_log.txt')))

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_digits
from softmax_regression import SoftmaxRegression
from neural_network import OneHiddenLayerNN
from optimizers import SGD
from train import train_softmax, train_nn
from track_b import run_track_b

def main():
    # 1. Load the fixed Digits benchmark data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "digits_data.npz")
    split_path = os.path.join(os.path.dirname(__file__), "..", "data", "digits_split_indices.npz")
    
    # Ensure relative paths work correctly
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_digits(data_path, split_path)
    
    print("Training models for Track B analysis...")
    
    # 2. Initialize and train Softmax Regression (SR)
    sr_model = SoftmaxRegression(d=64, k=10, lam=1e-4, seed=0)
    train_softmax(sr_model, SGD(lr=0.05), X_tr, y_tr, X_val, y_val)
    
    # 3. Initialize and train Neural Network (NN)
    nn_model = OneHiddenLayerNN(d=64, hidden=32, k=10, lam=1e-4, seed=0)
    train_nn(nn_model, SGD(lr=0.05), X_tr, y_tr, X_val, y_val)
    
    # 4. Run the advanced analysis
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    run_track_b(sr_model, nn_model, X_te, y_te, figures_dir=figures_dir)

if __name__ == "__main__":
    main()
