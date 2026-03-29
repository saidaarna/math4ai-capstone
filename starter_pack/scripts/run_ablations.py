#!/usr/bin/env python3
"""
Run ablation studies (capacity + optimizer) and statistical analysis (5 seeds).
Usage: python scripts/run_ablations.py
"""
import sys
import os

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ablations_and_stats import run_ablation_studies, run_statistical_analysis

if __name__ == "__main__":
    run_ablation_studies()
    run_statistical_analysis()
