# Weights & Biases (W&B) Integration for Qwen LoRA Trainer

This document explains how to use the newly implemented W&B tracking for monitoring your Qwen LoRA training progress.

## Features

The W&B integration provides comprehensive tracking of:

- **Per-step metrics:**
  - Training loss
  - Learning rate
  - Global step
  - Epoch progress

- **Per-epoch metrics:**
  - Average epoch loss
  - Epoch completion status
  - Training progress percentage

- **Training configuration:**
  - Model architecture details
  - Hyperparameters
  - Training settings

## Setup

### 1. Install W&B

```bash
pip install wandb
```

### 2. Set your W&B API key

```bash
export WANDB_API_KEY=your_api_key_here
```

You can get your API key from [wandb.ai/settings](https://wandb.ai/settings).

### 3. Configure W&B settings

Edit your training config file (e.g., `train_configs/train_lora_4090.yaml`):

```yaml
report_to: wandb
wandb_project_name: qwen-lora-training
wandb_run_name: lora-4090-run
wandb_entity: your_username  # Optional: set to your W&B username/team
wandb_tags: ["lora", "qwen", "text-to-image", "4090"]  # Optional: add tags
```

## Usage

### Basic Training

```bash
python train.py --config train_configs/train_lora_4090.yaml
```

### Test W&B Integration

Before running training, test that everything is set up correctly:

```bash
python test_wandb_integration.py
```

## What Gets Tracked

### Metrics per Training Step

- `train_loss`: Current training loss
- `learning_rate`: Current learning rate
- `epoch`: Current epoch number
- `epoch_progress`: Progress within current epoch (0.0 to 1.0)
- `global_step`: Total training steps completed

### Metrics per Epoch

- `epoch_loss`: Average loss for the completed epoch
- `epoch`: Epoch number
- `epoch_complete`: Boolean indicating epoch completion

### Final Training Metrics

- `training_complete`: Boolean indicating training completion
- `final_global_step`: Total steps completed
- `final_epoch`: Final epoch reached
- `final_learning_rate`: Final learning rate

### Configuration Tracking

The following training configuration is automatically logged:

- Model architecture and pretrained model path
- Learning rate and scheduler settings
- Batch sizes and gradient accumulation
- Mixed precision settings
- LoRA rank and other training parameters

## W&B Dashboard

Once training starts, you'll see:

1. **Real-time training curves** for loss and learning rate
2. **Epoch-by-epoch progress** with clear epoch boundaries
3. **Training configuration** in the run summary
4. **System metrics** (GPU usage, memory, etc.) if available

## Customization

### Change Project Name

```yaml
wandb_project_name: my-custom-project
```

### Add Run Tags

```yaml
wandb_tags: ["custom-tag", "experiment-1", "high-lr"]
```

### Set Entity/Team

```yaml
wandb_entity: my-team-name
```

### Disable W&B

To disable W&B tracking, set:

```yaml
report_to: null
```

## Troubleshooting

### W&B Not Logging

1. Check that `report_to: wandb` is set in your config
2. Verify your W&B API key is set: `echo $WANDB_API_KEY`
3. Ensure you have internet connectivity

### Missing Metrics

1. Check that training is actually running (look for progress bars)
2. Verify that the training script is using the updated version
3. Check console logs for any error messages

### API Key Issues

1. Verify your API key is correct
2. Check that your W&B account is active
3. Try logging in manually: `wandb login`

## Example W&B Run

A typical W&B run will show:

- **Overview tab**: Training summary and configuration
- **Charts tab**: Loss and learning rate curves
- **System tab**: GPU and memory usage (if available)
- **Files tab**: Model checkpoints and logs

## Best Practices

1. **Use descriptive run names** to easily identify experiments
2. **Add meaningful tags** for better organization
3. **Monitor training curves** to detect overfitting or learning issues
4. **Compare runs** to find optimal hyperparameters
5. **Save important checkpoints** and note them in W&B

## Support

If you encounter issues:

1. Check the console output for error messages
2. Verify your W&B account and API key
3. Test the integration with `test_wandb_integration.py`
4. Check the W&B documentation at [docs.wandb.ai](https://docs.wandb.ai)

## Example Output

```
🧪 Testing W&B Integration for Qwen LoRA Trainer
==================================================

🔍 Running: W&B Package Import
✅ wandb package can be imported
   Version: 0.15.12

🔍 Running: Accelerate Import
✅ accelerate package can be imported

🔍 Running: W&B Configuration
✅ Config file loaded successfully
✅ report_to is set to 'wandb'
✅ wandb_project_name: qwen-lora-training
✅ wandb_run_name: lora-4090-run

==================================================
📊 Test Results: 3/3 tests passed
🎉 All tests passed! W&B integration is ready.

📝 Next steps:
   1. Set your W&B API key: export WANDB_API_KEY=your_key_here
   2. Run training: python train.py --config train_configs/train_lora_4090.yaml
   3. Check your W&B dashboard for training progress
```