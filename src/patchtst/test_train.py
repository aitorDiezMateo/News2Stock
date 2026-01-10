"""
Quick Test Script for PatchTST
===============================
Runs a quick training test with reduced epochs to verify everything works.

Usage:
    python test_train.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from patchtst.config import Config
from patchtst.main import main

# Override config for quick testing
Config.NUM_EPOCHS = 5
Config.WARMUP_EPOCHS = 1
Config.BATCH_SIZE = 32

print("\n" + "="*70)
print("QUICK TEST MODE")
print("="*70)
print("\nModified Configuration:")
print(f"  - Epochs: {Config.NUM_EPOCHS} (reduced from 50)")
print(f"  - Warmup: {Config.WARMUP_EPOCHS} (reduced from 5)")
print(f"  - Batch size: {Config.BATCH_SIZE} (reduced from 64)")
print("="*70 + "\n")

# Run training
main()
