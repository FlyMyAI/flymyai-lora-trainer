#!/usr/bin/env python3
"""
Test script to verify LoRA checkpoint loading functionality.
This script tests that:
1. LoRA weights are loaded into the adapter modules
2. Base model weights remain unchanged
3. Only LoRA parameters are updated
"""

import os
import torch
import tempfile
import shutil
import logging
from diffusers import QwenImageTransformer2DModel
from peft import LoraConfig
import safetensors.torch

# Set up basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_checkpoint(model, checkpoint_dir):
    """Create a test checkpoint with some dummy LoRA weights."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create dummy LoRA weights (simulating what would be saved during training)
    lora_state_dict = {}

    # Get the current LoRA weights from the model
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_state_dict[name] = param.data.clone()

    # Add some dummy LoRA weights if none exist
    if not lora_state_dict:
        # Create some dummy LoRA weights for testing
        # Use smaller dimensions to avoid memory issues
        lora_state_dict = {
            'transformer_blocks.0.attn1.to_q.lora_A.weight': torch.randn(16, 128),
            'transformer_blocks.0.attn1.to_q.lora_B.weight': torch.randn(128, 16),
            'transformer_blocks.0.attn1.to_k.lora_A.weight': torch.randn(16, 128),
            'transformer_blocks.0.attn1.to_k.lora_B.weight': torch.randn(128, 16),
            'transformer_blocks.0.attn1.to_v.lora_A.weight': torch.randn(16, 128),
            'transformer_blocks.0.attn1.to_v.lora_B.weight': torch.randn(128, 16),
        }

    # Save the LoRA weights
    checkpoint_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
    safetensors.torch.save_file(lora_state_dict, checkpoint_path)

    return checkpoint_path

def test_lora_checkpoint_loading():
    """Test that LoRA checkpoint loading works correctly."""
    print("Testing LoRA checkpoint loading...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        try:
            # Load a base model
            print("Loading base model...")
            model = QwenImageTransformer2DModel.from_pretrained(
                "Qwen/Qwen-Image",
                subfolder="transformer",
                torch_dtype=torch.float32
            )

            # Store original base model weights for comparison
            # We need to store them before adding the LoRA adapter
            original_weights = {}
            for name, param in model.named_parameters():
                original_weights[name] = param.data.clone()

            print(f"Stored {len(original_weights)} base model parameters")

            # Add LoRA adapter
            print("Adding LoRA adapter...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            model.add_adapter(lora_config)

            # Get initial LoRA weights
            initial_lora_weights = {}
            for name, param in model.named_parameters():
                if 'lora' in name:
                    initial_lora_weights[name] = param.data.clone()

            print(f"Initial LoRA parameters: {len(initial_lora_weights)}")

            # Create a test checkpoint
            checkpoint_dir = os.path.join(temp_dir, "checkpoint-1000")
            checkpoint_path = create_test_checkpoint(model, checkpoint_dir)
            print(f"Created test checkpoint at: {checkpoint_path}")

            # Load the checkpoint using our function
            print("Loading checkpoint...")
            from train_4090 import load_checkpoint

            # Create a mock accelerator object
            class MockAccelerator:
                def __init__(self):
                    self.device = torch.device('cpu')

            mock_accelerator = MockAccelerator()

            # Load the checkpoint
            step = load_checkpoint(model, checkpoint_dir, mock_accelerator)
            print(f"Loaded checkpoint at step: {step}")

            # Verify that base model weights are unchanged
            print("Verifying base model weights are unchanged...")
            base_weights_changed = 0
            for name, param in model.named_parameters():
                if 'lora' not in name:
                    if not torch.allclose(param.data, original_weights[name]):
                        base_weights_changed += 1
                        print(f"  WARNING: Base weight changed: {name}")

            if base_weights_changed == 0:
                print("  ✓ All base model weights remain unchanged")
            else:
                print(f"  ✗ {base_weights_changed} base model weights were changed!")

            # Verify that LoRA weights were loaded
            print("Verifying LoRA weights were loaded...")
            lora_weights_loaded = 0
            for name, param in model.named_parameters():
                if 'lora' in name:
                    if not torch.allclose(param.data, initial_lora_weights[name]):
                        lora_weights_loaded += 1

            if lora_weights_loaded > 0:
                print(f"  ✓ {lora_weights_loaded} LoRA weights were updated")
            else:
                print("  ✗ No LoRA weights were updated!")

            # Check if the checkpoint file exists and has content
            if os.path.exists(checkpoint_path):
                checkpoint_size = os.path.getsize(checkpoint_path)
                print(f"  ✓ Checkpoint file exists with size: {checkpoint_size} bytes")
            else:
                print("  ✗ Checkpoint file not found!")

            return base_weights_changed == 0 and lora_weights_loaded > 0

        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        success = test_lora_checkpoint_loading()
        if success:
            print("\n🎉 All tests passed! LoRA checkpoint loading works correctly.")
        else:
            print("\n❌ Some tests failed. Check the output above.")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()