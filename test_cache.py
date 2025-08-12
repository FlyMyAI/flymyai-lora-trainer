#!/usr/bin/env python3
"""
Test script for the quantized model cache system.
This script demonstrates how to use the cache functionality.
"""

import os
import sys
import torch
from pathlib import Path

# Add the current directory to Python path to import from train_4090.py
sys.path.append('.')

# Import cache functions from train_4090.py
from train_4090 import (
    get_quantized_model_cache_path,
    save_quantized_model,
    load_quantized_model,
    validate_cache_integrity,
    get_cache_info,
    clear_old_cache_files,
    cleanup_corrupted_cache
)

def test_cache_system():
    """Test the cache system with a dummy model."""
    print("Testing quantized model cache system...")

    # Set cache directory
    os.environ['QWEN_CACHE'] = './test_cache'

    # Create a dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    model = DummyModel()

    # Test cache path generation
    cache_path = get_quantized_model_cache_path(
        "test_model_path",
        "float32",
        16
    )
    print(f"Cache path: {cache_path}")

    # Test cache info
    cache_dir = os.environ.get('QWEN_CACHE', './test_cache')
    cache_info = get_cache_info(cache_dir)
    print(f"Cache info: {cache_info}")

    # Test saving to cache
    print("Saving model to cache...")
    success = save_quantized_model(model, cache_path)
    print(f"Save successful: {success}")

    # Test cache validation
    print("Validating cache integrity...")
    is_valid = validate_cache_integrity(cache_path)
    print(f"Cache valid: {is_valid}")

    # Test loading from cache
    print("Loading model from cache...")
    new_model = DummyModel()
    load_success = load_quantized_model(new_model, cache_path)
    print(f"Load successful: {load_success}")

    # Test cache info after saving
    cache_info_after = get_cache_info(cache_dir)
    print(f"Cache info after saving: {cache_info_after}")

    # Test cleanup functions
    print("Testing cleanup functions...")
    corrupted_count = cleanup_corrupted_cache(cache_dir)
    print(f"Corrupted files cleaned: {corrupted_count}")

    # Test old cache cleanup (should not remove our test file)
    clear_old_cache_files(cache_dir, max_age_days=0)

    # Final cache info
    final_cache_info = get_cache_info(cache_dir)
    print(f"Final cache info: {final_cache_info}")

    print("Cache system test completed!")

if __name__ == "__main__":
    test_cache_system()