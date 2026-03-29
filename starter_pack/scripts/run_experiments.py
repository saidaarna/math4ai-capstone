#!/usr/bin/env python3
"""
Run the three core experiments (Gaussian / Moons / Digits).
Usage: python scripts/run_experiments.py
"""
import sys
import os

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments import run_experiments

if __name__ == "__main__":
    run_experiments()
