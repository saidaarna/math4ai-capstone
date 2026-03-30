#!/usr/bin/env python3
"""
Run sanity / gradient checks on the model implementations.
Usage: python scripts/run_sanity_checks.py
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

from sanity_checks import run_sanity_checks

if __name__ == "__main__":
    run_sanity_checks()
