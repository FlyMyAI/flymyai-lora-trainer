# Validation Functionality for LoRA Training

This document explains how to use the new validation functionality that has been added to both `train.py` and `train_4090.py` scripts.

## Overview

The validation functionality allows you to:
- Monitor model performance on a separate validation dataset during training
- Track validation loss at regular intervals (checkpoints and epochs)
- Log validation metrics to both console and Weights & Biases (W&B)
- Use different batch sizes and resolutions for validation vs. training

## Configuration

Add a `validation_config` block to your training configuration file:

```yaml
# Training data configuration
data_config:
  img_dir: "./training_data"
  img_size: 1024
  train_batch_size: 1

# Validation configuration
validation_config:
  img_dir: "./validation_data"  # Directory for validation images/text
  img_size: 1024                # Resolution for validation images
  train_batch_size: 1           # Batch size for validation
```

## Validation Dataset Structure

Your validation dataset should follow the same structure as your training dataset:

```
validation_data/
├── image1.png
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

- **Images**: `.png` or `.jpg` files
- **Text files**: Corresponding `.txt` files with the same base name containing prompts

## When Validation Runs

Validation loss is computed and logged at:

1. **Checkpoint saves**: Every `checkpointing_steps` (e.g., every 100 steps)
2. **Epoch completion**: At the end of each training epoch

## Logged Metrics

### Console Output
- Validation loss at each checkpoint: `Validation loss at step X: 0.123456`
- Validation loss at each epoch: `Epoch X validation loss: 0.123456`
- Epoch summaries including validation loss

### W&B Tracking
- `validation_loss`: Computed at checkpoints
- `epoch_validation_loss`: Computed at end of each epoch
- Both metrics include corresponding `global_step` and `epoch` numbers

## Example Configuration

See `example_config_with_validation.yaml` for a complete example configuration.

## Key Features

### Flexible Configuration
- **Different batch sizes**: Training and validation can use different batch sizes
- **Different resolutions**: Validation images can be processed at different resolutions than training
- **Independent preprocessing**: Validation can use different image/text preprocessing if needed

### Memory Efficient
- Validation embeddings are precomputed if `precompute_*_embeddings` is enabled
- Validation runs with `torch.no_grad()` to save memory
- Separate VAE and text encoding pipelines for validation to avoid conflicts

### Comprehensive Logging
- All validation metrics are logged to W&B with proper step tracking
- Console output shows validation progress clearly
- Epoch summaries include both training and validation metrics

## Usage Examples

### Basic Validation Setup
```bash
# Train with validation
python train.py --config config_with_validation.yaml
```

### Monitor Validation in W&B
1. Start training with validation enabled
2. Open your W&B project
3. View validation loss curves alongside training loss
4. Compare training vs. validation performance

### Custom Validation Parameters
```yaml
validation_config:
  img_dir: "./small_validation_set"
  img_size: 512        # Smaller images for faster validation
  train_batch_size: 2  # Larger batch size for validation
```

## Performance Considerations

- **Validation frequency**: More frequent validation = more accurate monitoring but slower training
- **Validation dataset size**: Larger validation sets give more reliable metrics but take longer to compute
- **Image resolution**: Smaller validation images process faster but may not capture all training details
- **Batch size**: Larger validation batch sizes can be more memory efficient

## Troubleshooting

### Common Issues

1. **Validation directory not found**: Ensure `validation_config.img_dir` points to a valid directory
2. **Memory issues**: Reduce validation batch size or image resolution
3. **Slow validation**: Consider using smaller validation images or fewer validation samples

### Debug Mode
Enable debug logging to see detailed validation setup:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Validation Schedules
You can modify the validation frequency by changing the checkpointing steps or adding custom validation triggers.

### Multiple Validation Sets
For more sophisticated validation, you can modify the code to support multiple validation datasets or different validation metrics.

### Validation Metrics
Currently, only loss is computed. You can extend the `compute_validation_loss` function to include additional metrics like:
- Perceptual similarity scores
- FID scores
- Custom image quality metrics

## Integration with Existing Workflows

The validation functionality is designed to be:
- **Non-intrusive**: Existing training scripts work without modification
- **Configurable**: Enable/disable via configuration
- **Compatible**: Works with all existing training features (LoRA, quantization, etc.)
- **Scalable**: Handles both small and large validation datasets efficiently