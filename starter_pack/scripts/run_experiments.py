#!/usr/bin/env python3
"""
Launcher for running all experiments (Gaussian / Moons / Digits)
Usage: python scripts/run_experiments.py
"""
import sys
import os

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments import run_experiments

if __name__ == "__main__":
    run_experiments()