#!/usr/bin/env python3
"""
Run sanity / gradient checks on the model implementations.
Usage: python scripts/run_sanity_checks.py
"""
import sys
import os

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sanity_checks import run_sanity_checks

if __name__ == "__main__":
    run_sanity_checks()
