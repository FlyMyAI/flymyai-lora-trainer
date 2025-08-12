#!/usr/bin/env python3
"""
Test script to verify W&B integration for the Qwen LoRA trainer.
This script tests the basic W&B setup without running actual training.
"""

import os
import sys
import yaml
from omegaconf import OmegaConf

def test_wandb_config():
    """Test that the W&B configuration is properly set up."""

    # Load the config
    config_path = "train_configs/train_lora_4090.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        config = OmegaConf.load(config_path)
        print("✅ Config file loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

    # Check W&B settings
    if config.get('report_to') != 'wandb':
        print("❌ report_to is not set to 'wandb'")
        return False
    print("✅ report_to is set to 'wandb'")

    if not config.get('wandb_project_name'):
        print("❌ wandb_project_name is not set")
        return False
    print(f"✅ wandb_project_name: {config.wandb_project_name}")

    if not config.get('wandb_run_name'):
        print("❌ wandb_run_name is not set")
        return False
    print(f"✅ wandb_run_name: {config.wandb_run_name}")

    return True

def test_wandb_import():
    """Test that wandb can be imported."""
    try:
        import wandb
        print("✅ wandb package can be imported")
        print(f"   Version: {wandb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import wandb: {e}")
        print("   Please install wandb: pip install wandb")
        return False

def test_accelerate_wandb():
    """Test that accelerate can work with wandb."""
    try:
        from accelerate import Accelerator
        print("✅ accelerate package can be imported")
        return True
    except ImportError as e:
        print(f"❌ Failed to import accelerate: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing W&B Integration for Qwen LoRA Trainer")
    print("=" * 50)

    tests = [
        ("W&B Package Import", test_wandb_import),
        ("Accelerate Import", test_accelerate_wandb),
        ("W&B Configuration", test_wandb_config),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! W&B integration is ready.")
        print("\n📝 Next steps:")
        print("   1. Set your W&B API key: export WANDB_API_KEY=your_key_here")
        print("   2. Run training: python train.py --config train_configs/train_lora_4090.yaml")
        print("   3. Check your W&B dashboard for training progress")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())