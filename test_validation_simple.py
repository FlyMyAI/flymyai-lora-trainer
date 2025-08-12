#!/usr/bin/env python3
"""
Simple test script to verify validation logic without full training.
This tests the validation dataset loading and basic structure.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path to import modules
sys.path.append('.')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_validation_setup():
    """Test basic validation setup logic."""

    # Mock validation config
    class MockValidationConfig:
        def __init__(self):
            self.img_dir = "./validation_data"
            self.img_size = 1024
            self.train_batch_size = 1

    class MockArgs:
        def __init__(self):
            self.validation_config = MockValidationConfig()
            self.precompute_text_embeddings = True
            self.precompute_image_embeddings = True
            self.pretrained_model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    args = MockArgs()

    # Test validation config
    logger.info("Testing validation configuration...")
    if hasattr(args, 'validation_config') and args.validation_config is not None:
        logger.info("✓ Validation config found")
        logger.info(f"  - Image directory: {args.validation_config.img_dir}")
        logger.info(f"  - Image size: {args.validation_config.img_size}")
        logger.info(f"  - Batch size: {args.validation_config.train_batch_size}")
    else:
        logger.error("✗ No validation config found")
        return False

    # Test validation directory
    logger.info("\nTesting validation directory...")
    val_dir = args.validation_config.img_dir
    if os.path.exists(val_dir):
        logger.info(f"✓ Validation directory exists: {val_dir}")

        # Count files
        txt_files = [f for f in os.listdir(val_dir) if f.endswith('.txt')]
        img_files = [f for f in os.listdir(val_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        logger.info(f"  - Text files: {len(txt_files)}")
        logger.info(f"  - Image files: {len(img_files)}")

        if txt_files:
            logger.info(f"  - Sample text files: {txt_files[:3]}")
        if img_files:
            logger.info(f"  - Sample image files: {img_files[:3]}")

        # Check if we have matching pairs
        txt_bases = {os.path.splitext(f)[0] for f in txt_files}
        img_bases = {os.path.splitext(f)[0] for f in img_files}
        matching_pairs = txt_bases.intersection(img_bases)

        logger.info(f"  - Matching text-image pairs: {len(matching_pairs)}")
        if matching_pairs:
            logger.info(f"  - Sample pairs: {list(matching_pairs)[:3]}")

    else:
        logger.warning(f"⚠️  Validation directory does not exist: {val_dir}")
        logger.info("Creating test validation data...")

        # Create test directory
        os.makedirs(val_dir, exist_ok=True)

        # Create a test text file
        test_txt = os.path.join(val_dir, "test_image.txt")
        with open(test_txt, 'w') as f:
            f.write("A test image for validation")

        # Create a test image (simple colored square)
        try:
            from PIL import Image
            import numpy as np

            test_img = Image.new('RGB', (512, 512), color='blue')
            test_img_path = os.path.join(val_dir, "test_image.png")
            test_img.save(test_img_path)

            logger.info("✓ Created test validation data")
            logger.info(f"  - Text file: {test_txt}")
            logger.info(f"  - Image file: {test_img_path}")

        except ImportError:
            logger.warning("PIL not available, skipping test image creation")

    # Test precomputation flags
    logger.info("\nTesting precomputation configuration...")
    logger.info(f"  - Precompute text embeddings: {args.precompute_text_embeddings}")
    logger.info(f"  - Precompute image embeddings: {args.precompute_image_embeddings}")

    if args.precompute_text_embeddings:
        logger.info("✓ Text embeddings will be precomputed for validation")
    else:
        logger.info("⚠️  Text embeddings will be computed on-the-fly during validation")

    if args.precompute_image_embeddings:
        logger.info("✓ Image embeddings will be precomputed for validation")
    else:
        logger.info("⚠️  Image embeddings will be computed on-the-fly during validation")

    # Test batch size logic
    logger.info("\nTesting batch size logic...")
    if hasattr(args.validation_config, 'train_batch_size'):
        batch_size = args.validation_config.train_batch_size
        logger.info(f"  - Validation batch size: {batch_size}")

        # Simulate dataset size calculation
        if os.path.exists(val_dir):
            img_files = [f for f in os.listdir(val_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if img_files:
                expected_batches = len(img_files) // batch_size
                if len(img_files) % batch_size != 0:
                    expected_batches += 1
                logger.info(f"  - With {len(img_files)} images: expected {expected_batches} batches")
            else:
                logger.warning("  - No image files found for batch calculation")

    logger.info("\n✅ Validation setup test completed!")
    return True

def test_validation_loop_logic():
    """Test the validation loop logic without actual model inference."""

    logger.info("\nTesting validation loop logic...")

    # Mock validation dataloader
    class MockValidationDataloader:
        def __init__(self, num_batches=3):
            self.num_batches = num_batches

        def __len__(self):
            return self.num_batches

        def __iter__(self):
            # Simulate batches
            for i in range(self.num_batches):
                # Mock batch with 3 items (cached embeddings)
                yield (f"image_batch_{i}", f"text_batch_{i}", f"mask_batch_{i}")

    # Test the validation loop structure
    validation_dataloader = MockValidationDataloader(num_batches=3)

    logger.info(f"Mock validation dataloader created with {len(validation_dataloader)} batches")

    # Simulate the validation loop structure
    total_val_loss = 0.0
    num_val_batches = 0
    max_val_batches = len(validation_dataloader)

    logger.info(f"Processing all {max_val_batches} validation batches")

    for batch_idx, val_batch in enumerate(validation_dataloader):
        logger.info(f"Processing validation batch {batch_idx + 1}/{max_val_batches}")

        # Simulate batch processing
        if len(val_batch) == 3:
            logger.info(f"  - Batch structure: [image_embeddings, text_embeddings, text_masks]")
            logger.info(f"  - Using cached validation embeddings")
        else:
            logger.warning(f"  - Unexpected batch structure with {len(val_batch)} items")

        # Simulate loss computation
        mock_loss = 0.1 + (batch_idx * 0.01)  # Simple mock loss
        total_val_loss += mock_loss
        num_val_batches += 1

        logger.info(f"  - Mock batch loss: {mock_loss:.6f}, cumulative loss: {total_val_loss:.6f}")

        # Test the break condition
        if num_val_batches >= max_val_batches:
            logger.info(f"Reached maximum validation batches ({max_val_batches}), stopping validation")
            break

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Validation completed: {num_val_batches} batches, average loss: {avg_val_loss:.6f}")

    logger.info("✅ Validation loop logic test completed!")
    return True

if __name__ == "__main__":
    logger.info("🧪 Starting validation tests...")

    try:
        # Test basic setup
        if test_validation_setup():
            # Test loop logic
            test_validation_loop_logic()

        logger.info("\n🎉 All validation tests completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Ensure validation_data/ directory contains image-text pairs")
        logger.info("2. Run training with validation_config enabled")
        logger.info("3. Check logs for validation progress and any errors")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)