import argparse
import copy
from copy import deepcopy
import json
import logging
import os
import shutil
import hashlib
import pickle
import glob

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.dataset import loader, image_resize
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from PIL import Image
import numpy as np
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb
logger = get_logger(__name__, log_level="INFO")
from diffusers.loaders import AttnProcsLayers
import gc
from custom_wandb_tracker import CustomWandbTracker


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    if not os.path.exists(output_dir):
        return None

    # Look for checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None

    # Extract step numbers and find the latest
    checkpoint_steps = []
    for checkpoint_dir in checkpoint_dirs:
        try:
            step = int(checkpoint_dir.split("-")[-1])
            checkpoint_steps.append((step, checkpoint_dir))
        except (ValueError, IndexError):
            continue

    if not checkpoint_steps:
        return None

    # Sort by step number and return the latest
    latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])
    return latest_checkpoint[1]


def load_checkpoint(flux_transformer, checkpoint_path, accelerator):
    """Load a LoRA checkpoint and return the global step.

    Note: This loads only the LoRA weights into the LoRA adapter modules,
    preserving the base model weights. The training will start from the
    beginning of the dataset but with the loaded LoRA weights.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} does not exist")
        return 0

    # Look for the safetensors file
    safetensors_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"LoRA weights file not found at {safetensors_path}")
        return 0

    try:
        # Load the LoRA weights
        print(f"Loading checkpoint from {safetensors_path}")

        # Load the safetensors file directly
        import safetensors.torch
        checkpoint_state_dict = safetensors.torch.load_file(safetensors_path)

        # Filter to only LoRA weights (keys containing 'lora')
        lora_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            if 'lora' in key:
                lora_state_dict[key] = value

        if not lora_state_dict:
            print("No LoRA weights found in checkpoint")
            return 0

        print(f"Found {len(lora_state_dict)} LoRA weight keys")

        # Load only the LoRA weights into the model
        # This preserves the base model weights and only updates LoRA parameters
        missing_keys, unexpected_keys = flux_transformer.load_state_dict(lora_state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys when loading LoRA weights: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys when loading LoRA weights: {unexpected_keys[:5]}...")

        # Extract the step number from the checkpoint path
        checkpoint_name = os.path.basename(checkpoint_path)
        if checkpoint_name.startswith("checkpoint-"):
            try:
                step = int(checkpoint_name.split("-")[-1])
                print(f"Successfully loaded LoRA checkpoint at step {step}")
                return step
            except ValueError:
                print("Could not parse step number from checkpoint name")
                return 0
        else:
            print("Checkpoint directory name format not recognized")
            return 0

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 0


def get_quantized_model_cache_path(model_path, weight_dtype, rank):
    """Generate a cache path for the quantized model based on model path, dtype, and rank."""
    cache_dir = os.environ.get('QWEN_CACHE', os.path.expanduser('~/.cache/qwen_quantized'))
    os.makedirs(cache_dir, exist_ok=True)

    # Create a hash of the model path and configuration
    config_str = f"{model_path}_{weight_dtype}_{rank}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    return os.path.join(cache_dir, f"quantized_model_{config_hash}.pkl")


def validate_cache_integrity(cache_path):
    """Validate that the cache file is not corrupted and can be loaded."""
    try:
        if not os.path.exists(cache_path):
            return False

        # Try to load and parse the cache file
        with open(cache_path, 'rb') as f:
            model_state = pickle.load(f)

        # Basic validation - check if it's a dict and has some content
        if not isinstance(model_state, dict) or len(model_state) == 0:
            logger.warning(f"Cache file {cache_path} is corrupted or empty")
            return False

        return True
    except Exception as e:
        logger.warning(f"Cache file {cache_path} is corrupted: {e}")
        return False


def save_quantized_model(model, cache_path):
    """Save the quantized model to cache."""
    try:
        # Save the model state dict
        model_state = model.state_dict()

        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(model_state, f)

        logger.info(f"Quantized model saved to cache: {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save quantized model to cache: {e}")
        return False


def load_quantized_model(model, cache_path):
    """Load the quantized model from cache."""
    try:
        if not os.path.exists(cache_path):
            return False

        # Load from cache
        with open(cache_path, 'rb') as f:
            model_state = pickle.load(f)

        # Check if the cached state dict is compatible
        model_keys = set(model.state_dict().keys())
        cache_keys = set(model_state.keys())

        if model_keys != cache_keys:
            logger.warning(f"Cache state dict has different keys. Expected: {len(model_keys)}, Got: {len(cache_keys)}")
            missing_keys = model_keys - cache_keys
            extra_keys = cache_keys - model_keys
            if missing_keys:
                logger.warning(f"Missing keys: {list(missing_keys)[:5]}...")
            if extra_keys:
                logger.warning(f"Extra keys: {list(extra_keys)[:5]}...")
            return False

        # Load the state dict
        model.load_state_dict(model_state)

        logger.info(f"Quantized model loaded from cache: {cache_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load quantized model from cache: {e}")
        return False


def clear_old_cache_files(cache_dir, max_age_days=7):
    """Clear old cache files to prevent disk space issues."""
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    logger.info(f"Removed old cache file: {filename}")
    except Exception as e:
        logger.warning(f"Failed to clear old cache files: {e}")


def get_cache_info(cache_dir):
    """Get information about the cache directory."""
    try:
        if not os.path.exists(cache_dir):
            return "Cache directory does not exist"

        files = os.listdir(cache_dir)
        total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in files if os.path.isfile(os.path.join(cache_dir, f)))

        return {
            "cache_dir": cache_dir,
            "file_count": len(files),
            "total_size_mb": total_size / (1024 * 1024)
        }
    except Exception as e:
        return f"Error getting cache info: {e}"


def cleanup_corrupted_cache(cache_dir):
    """Remove corrupted cache files to prevent future loading failures."""
    try:
        corrupted_files = []
        for filename in os.listdir(cache_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(cache_dir, filename)
                if not validate_cache_integrity(file_path):
                    corrupted_files.append(filename)
                    os.remove(file_path)

        if corrupted_files:
            logger.info(f"Cleaned up {len(corrupted_files)} corrupted cache files")
        return len(corrupted_files)
    except Exception as e:
        logger.warning(f"Failed to cleanup corrupted cache: {e}")
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10):
        self.data = torch.randn(num_samples, input_dim)    # random features
        self.labels = torch.randint(0, 2, (num_samples,))  # random labels: 0 or 1

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


def lora_processors(model):
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
            print(name)
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def compute_validation_loss(flux_transformer, vae, text_encoding_pipeline, noise_scheduler_copy,
                           validation_dataloader, accelerator, weight_dtype, get_sigmas):
    """Compute validation loss on the validation dataset."""
    flux_transformer.eval()
    total_val_loss = 0.0
    num_val_batches = 0

    logger.info(f"Starting validation on {len(validation_dataloader)} batches")

    # Calculate actual validation batches based on validation dataset size
    # This prevents infinite loops while processing the actual validation data
    max_val_batches = len(validation_dataloader)
    logger.info(f"Processing all {max_val_batches} validation batches")

    with torch.no_grad():
        for batch_idx, val_batch in enumerate(validation_dataloader):
            logger.info(f"Processing validation batch {batch_idx + 1}/{max_val_batches}")

            # Break after processing max_val_batches to prevent infinite loops
            if num_val_batches >= max_val_batches:
                logger.info(f"Reached maximum validation batches ({max_val_batches}), stopping validation")
                break

            if num_val_batches == 0:  # Log batch structure only once
                logger.info(f"Validation batch structure: {len(val_batch)} items")
                if len(val_batch) == 3:
                    logger.info("Batch contains: [image_embeddings, text_embeddings, text_masks]")
                elif len(val_batch) == 2:
                    logger.info("Batch contains: [raw_images, raw_texts]")
                else:
                    logger.warning(f"Unexpected batch structure with {len(val_batch)} items")

                # Log batch shapes for debugging
                if len(val_batch) == 3:
                    logger.info(f"Image embeddings shape: {val_batch[0].shape if hasattr(val_batch[0], 'shape') else 'N/A'}")
                    logger.info(f"Text embeddings shape: {val_batch[1].shape if hasattr(val_batch[1], 'shape') else 'N/A'}")
                    logger.info(f"Text masks shape: {val_batch[2].shape if hasattr(val_batch[2], 'shape') else 'N/A'}")
                elif len(val_batch) == 2:
                    logger.info(f"Raw images shape: {val_batch[0].shape if hasattr(val_batch[0], 'shape') else 'N/A'}")
                    logger.info(f"Raw texts type: {type(val_batch[1])}")

            if len(val_batch) == 3:  # With cached embeddings
                img, prompt_embeds, prompt_embeds_mask = val_batch
                prompt_embeds, prompt_embeds_mask = prompt_embeds.to(dtype=weight_dtype).to(accelerator.device), prompt_embeds_mask.to(dtype=torch.int32).to(accelerator.device)
                if num_val_batches == 0:  # Log only once
                    logger.info("Using cached validation embeddings")
            else:  # Without cached embeddings
                img, prompts = val_batch
                if num_val_batches == 0:  # Log only once
                    logger.info("Computing validation embeddings on-the-fly")
                if text_encoding_pipeline is None:
                    raise ValueError("text_encoding_pipeline is required when not using cached embeddings")
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    prompt=prompts,
                    device=accelerator.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )

            # Handle image processing
            if isinstance(img, torch.Tensor):  # Cached embeddings
                pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)
                if num_val_batches == 0: # Log only once
                    logger.info("Using cached validation image embeddings")
            else:  # Raw images
                if num_val_batches == 0: # Log only once
                    logger.info("Computing validation image embeddings on-the-fly")
                pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                pixel_values = pixel_values.unsqueeze(2)
                pixel_latents = vae.encode(pixel_values).latent_dist.sample()

            logger.debug(f"Processing batch {batch_idx + 1}: pixel_latents shape = {pixel_latents.shape}")

            pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, 1, vae.config.z_dim, 1, 1)
                .to(pixel_latents.device, pixel_latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                pixel_latents.device, pixel_latents.dtype
            )
            pixel_latents = (pixel_latents - latents_mean) * latents_std

            bsz = pixel_latents.shape[0]
            noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
            u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

            sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            packed_noisy_model_input = QwenImagePipeline._pack_latents(
                noisy_model_input,
                bsz,
                noisy_model_input.shape[2],
                noisy_model_input.shape[3],
                noisy_model_input.shape[4],
            )

            img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz

            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

            model_pred = flux_transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

            vae_scale_factor = 2 ** len(vae.temperal_downsample)
            model_pred = QwenImagePipeline._unpack_latents(
                model_pred,
                height=noisy_model_input.shape[3] * vae_scale_factor,
                width=noisy_model_input.shape[4] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )

            weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
            target = noise - pixel_latents
            target = target.permute(0, 2, 1, 3, 4)
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            total_val_loss += loss.item()
            num_val_batches += 1

            logger.info(f"Batch {batch_idx + 1} loss: {loss.item():.6f}, average loss: {total_val_loss / num_val_batches:.6f}")

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Validation completed: {num_val_batches} batches, average loss: {avg_val_loss:.6f}")
    flux_transformer.train()
    return avg_val_loss


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

        # Initialize cache system
    cache_dir = os.environ.get('QWEN_CACHE', os.path.expanduser('~/.cache/qwen_quantized'))
    if args.quantize:
        logger.info(f"Using quantized model cache directory: {cache_dir}")
        cache_info = get_cache_info(cache_dir)
        logger.info(f"Cache info: {cache_info}")

        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(cache_dir)
            free_gb = free / (1024**3)
            logger.info(f"Available disk space: {free_gb:.2f} GB")

            if free_gb < 1.0:  # Less than 1GB free
                logger.warning("Low disk space detected. Consider clearing old cache files.")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        # Optionally clear old cache files (uncomment if needed)
        # clear_old_cache_files(cache_dir, max_age_days=7)

        # Clean up corrupted cache files automatically
        corrupted_count = cleanup_corrupted_cache(cache_dir)
        if corrupted_count > 0:
            logger.info(f"Automatically cleaned up {corrupted_count} corrupted cache files")

    # Initialize W&B tracking with more comprehensive config
    wandb_config = {
        key: value for key, value in args.items() if key not in ['wandb_run_name', 'wandb_project_name', 'wandb_entity', 'wandb_tags']
        # "total_epochs": total_epochs,
        # "steps_per_epoch": steps_per_epoch,
        # "optimizer": getattr(args, 'optimizer', 'adam'),
    }

    # Add optimizer-specific parameters to W&B config
    if hasattr(args, 'optimizer_args'):
        wandb_config["optimizer_args"] = args.optimizer_args

    # Add validation config to W&B if available
    if hasattr(args, 'validation_config') and args.validation_config is not None:
        wandb_config["validation_batch_size"] = args.validation_config.get("train_batch_size", "N/A")
        wandb_config["validation_img_size"] = args.validation_config.get("img_size", "N/A")
        wandb_config["validation_img_dir"] = args.validation_config.get("img_dir", "N/A")

    # Initialize custom W&B tracker
    wandb_tracker = CustomWandbTracker(
        run_name=args.wandb_run_name,
        project_name=args.wandb_project_name,
        entity=args.wandb_entity,
        tags=args.wandb_tags,
        config=wandb_config,
    )

    wandb_tracker.store_init_configuration(wandb_config)


    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=[wandb_tracker],
        project_config=accelerator_project_config,
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_project_name, config=wandb_config)
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)
    cached_text_embeddings = None
    if args.precompute_text_embeddings:
        with torch.no_grad():
            cached_text_embeddings = {}
            for txt in tqdm([i for i in os.listdir(args.data_config.img_dir) if ".txt" in i]):
                txt_path = os.path.join(args.data_config.img_dir, txt)
                prompt = open(txt_path).read()
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    prompt=[prompt],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                cached_text_embeddings[txt] = {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}
            # compute empty embedding
            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                prompt=[' '],
                device=text_encoding_pipeline.device,
                num_images_per_prompt=1,
                max_sequence_length=1024,
            )
            cached_text_embeddings['empty_embedding'] = {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}
        text_encoding_pipeline.to("cpu")
        torch.cuda.empty_cache()
    # Delete text_encoding_pipeline since validation will handle its own needs
    # If validation needs text encoding and we don't have cached embeddings, it will create a new one
    del text_encoding_pipeline
    gc.collect()

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    cached_image_embeddings = None
    if args.precompute_image_embeddings:
        cached_image_embeddings = {}
        with torch.no_grad():
            for img_name in tqdm([i for i in os.listdir(args.data_config.img_dir) if ".png" in i or ".jpg" in i]):
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                img = image_resize(img, args.data_config.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1).unsqueeze(0)
                pixel_values = img.unsqueeze(2)
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)

                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                cached_image_embeddings[img_name] = pixel_latents
        vae.to('cpu')
        torch.cuda.empty_cache()
    #del vae
    #gc.collect()
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",    )

    if args.quantize:
        torch_dtype = weight_dtype
        device = accelerator.device

        # Try to load from cache first
        cache_path = get_quantized_model_cache_path(
            args.pretrained_model_name_or_path,
            str(weight_dtype),
            args.rank
        )

        if validate_cache_integrity(cache_path) and load_quantized_model(flux_transformer, cache_path):
            logger.info("Successfully loaded quantized model from cache")
        else:
            logger.info("Cache miss - quantizing model from scratch")
            all_blocks = list(flux_transformer.transformer_blocks)
            print(f"Quantizing {len(all_blocks)} blocks")
            for block in tqdm(all_blocks):
                block.to(device, dtype=torch_dtype)
                quantize(block, weights=qfloat8)
                freeze(block)
                block.to('cpu')
            flux_transformer.to(device, dtype=torch_dtype)
            quantize(flux_transformer, weights=qfloat8)
            freeze(flux_transformer)

            # Save to cache for future use
            save_quantized_model(flux_transformer, cache_path)

        #quantize(flux_transformer, weights=qint8, activations=qint8)
        #freeze(flux_transformer)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    flux_transformer.to(accelerator.device)
    #flux_transformer.add_adapter(lora_config)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.add_adapter(lora_config)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    flux_transformer.requires_grad_(False)

    flux_transformer.train()

    # Set up optimizer based on configuration
    optimizer_type = getattr(args, 'optimizer', 'adam').lower()

    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True
            print(n)
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))
    flux_transformer.enable_gradient_checkpointing()

    # Get optimizer arguments from config
    optimizer_args = getattr(args, 'optimizer_args', {})

    if args.adam8bit:
        # For 8-bit Adam, we need to handle the case where we might be using different optimizers
        if optimizer_type == 'adam':
            optimizer = bnb.optim.Adam8bit(
                lora_layers,
                lr=args.learning_rate,
                betas=tuple(optimizer_args.get('betas', [0.9, 0.999])),
                weight_decay=optimizer_args.get('weight_decay', 0.01),
                eps=optimizer_args.get('epsilon', 1e-8),
            )
        else:
            logger.warning("8-bit optimization only supported for Adam optimizer. Falling back to regular optimizer.")
            args.adam8bit = False
            optimizer_type = 'adam'

    if not args.adam8bit:
        if optimizer_type == 'adam':
            optimizer_cls = torch.optim.AdamW
            optimizer = optimizer_cls(
                lora_layers,
                lr=args.learning_rate,
                betas=tuple(optimizer_args.get('betas', [0.9, 0.999])),
                weight_decay=optimizer_args.get('weight_decay', 0.01),
                eps=optimizer_args.get('epsilon', 1e-8),
            )
        elif optimizer_type == 'adafactor':
            try:
                from transformers import Adafactor
                optimizer = Adafactor(
                    lora_layers,
                    lr=args.learning_rate,
                    betas=tuple(optimizer_args.get('betas', [0.9, 0.999])),
                    weight_decay=optimizer_args.get('weight_decay', 0.01),
                    eps=optimizer_args.get('epsilon', 1e-8),
                    scale_parameter=optimizer_args.get('scale_parameter', True),
                    relative_step=optimizer_args.get('relative_step', True),
                    warmup_init=optimizer_args.get('warmup_init', True),
                )
            except ImportError:
                logger.warning("Adafactor not available, falling back to AdamW")
                optimizer_type = 'adam'
                optimizer_cls = torch.optim.AdamW
                optimizer = optimizer_cls(
                    lora_layers,
                    lr=args.learning_rate,
                    betas=tuple(optimizer_args.get('betas', [0.9, 0.999])),
                    weight_decay=optimizer_args.get('weight_decay', 0.01),
                    eps=optimizer_args.get('epsilon', 1e-8),
                )
        elif optimizer_type == 'prodigy':
            try:
                from prodigyopt import Prodigy
                # Prodigy uses three beta values
                betas = optimizer_args.get('betas', [0.9, 0.999])
                if len(betas) == 2:
                    betas = betas + [optimizer_args.get('beta3', 0.999)]

                optimizer = Prodigy(
                    lora_layers,
                    lr=args.learning_rate,
                    betas=tuple(betas),
                    weight_decay=optimizer_args.get('weight_decay', 0.01),
                    eps=optimizer_args.get('epsilon', 1e-8),
                    use_bias_correction=optimizer_args.get('use_bias_correction', True),
                    safeguard_warmup=optimizer_args.get('safeguard_warmup', True),
                    d_coef=optimizer_args.get('d_coef', 0.1),
                )
            except ImportError:
                raise ImportError("Prodigy not available, please install it with `pip install prodigyopt`")
        else:
            logger.warning(f"Unknown optimizer type '{optimizer_type}', falling back to AdamW")
            optimizer_type = 'adam'
            optimizer_cls = torch.optim.AdamW
            optimizer = optimizer_cls(
                lora_layers,
                lr=args.learning_rate,
                betas=tuple(optimizer_args.get('betas', [0.9, 0.999])),
                weight_decay=optimizer_args.get('weight_decay', 0.01),
                eps=optimizer_args.get('epsilon', 1e-8),
            )

    logger.info(f"Using {optimizer_type.title()} optimizer")
    train_dataloader = loader(cached_text_embeddings=cached_text_embeddings, cached_image_embeddings=cached_image_embeddings, **args.data_config)

    # Setup validation dataset if validation_config is provided
    validation_dataloader = None
    if hasattr(args, 'validation_config') and args.validation_config is not None:
        logger.info("Setting up validation dataset...")
        # For validation, we need to handle both cached and non-cached cases
        val_cached_text_embeddings = None
        val_cached_image_embeddings = None

        if args.precompute_text_embeddings and hasattr(args.validation_config, 'img_dir'):
            # Precompute validation text embeddings
            logger.info("Precomputing validation text embeddings...")
            val_cached_text_embeddings = {}
            text_encoding_pipeline_val = QwenImagePipeline.from_pretrained(
                args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
            )
            text_encoding_pipeline_val.to(accelerator.device)
            with torch.no_grad():
                txt_files = [i for i in os.listdir(args.validation_config.img_dir) if ".txt" in i]
                logger.info(f"Found {len(txt_files)} text files for validation")
                for txt in tqdm(txt_files, desc="Precomputing validation text embeddings"):
                    txt_path = os.path.join(args.validation_config.img_dir, txt)
                    prompt = open(txt_path).read()
                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline_val.encode_prompt(
                        prompt=[prompt],
                        device=text_encoding_pipeline_val.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    val_cached_text_embeddings[txt] = {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}
                # compute empty embedding
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline_val.encode_prompt(
                    prompt=[' '],
                    device=text_encoding_pipeline_val.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                val_cached_text_embeddings['empty_embedding'] = {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}
            text_encoding_pipeline_val.to("cpu")
            torch.cuda.empty_cache()
            del text_encoding_pipeline_val
            gc.collect()
            logger.info(f"Precomputed {len(val_cached_text_embeddings)} validation text embeddings")

        if args.precompute_image_embeddings and hasattr(args.validation_config, 'img_dir'):
            # Precompute validation image embeddings
            logger.info("Precomputing validation image embeddings...")
            val_cached_image_embeddings = {}
            vae_val = AutoencoderKLQwenImage.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
            )
            vae_val.to(accelerator.device, dtype=weight_dtype)
            with torch.no_grad():
                img_files = [i for i in os.listdir(args.validation_config.img_dir) if ".png" in i or ".jpg" in i]
                logger.info(f"Found {len(img_files)} image files for validation")
                for img_name in tqdm(img_files, desc="Precomputing validation image embeddings"):
                    img = Image.open(os.path.join(args.validation_config.img_dir, img_name)).convert('RGB')
                    img = image_resize(img, args.validation_config.img_size)
                    w, h = img.size
                    new_w = (w // 32) * 32
                    new_h = (h // 32) * 32
                    img = img.resize((new_w, new_h))
                    img = torch.from_numpy((np.array(img) / 127.5) - 1)
                    img = img.permute(2, 0, 1).unsqueeze(0)
                    pixel_values = img.unsqueeze(2)
                    pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)

                    pixel_latents = vae_val.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                    val_cached_image_embeddings[img_name] = pixel_latents
            vae_val.to('cpu')
            torch.cuda.empty_cache()
            del vae_val
            gc.collect()
            logger.info(f"Precomputed {len(val_cached_image_embeddings)} validation image embeddings")

        validation_dataloader = loader(
            cached_text_embeddings=val_cached_text_embeddings,
            cached_image_embeddings=val_cached_image_embeddings,
            **args.validation_config
        )
        logger.info(f"Validation dataset loaded with {len(validation_dataloader)} batches")

        # Additional validation dataset debugging info
        if hasattr(validation_dataloader, 'dataset'):
            logger.info(f"Validation dataset has {len(validation_dataloader.dataset)} samples")
            logger.info(f"Validation batch size: {args.validation_config.train_batch_size}")
            logger.info(f"Expected batches: {len(validation_dataloader.dataset) // args.validation_config.train_batch_size}")
        else:
            logger.warning("Validation dataloader has no dataset attribute")

        # Log validation config details
        logger.info(f"Validation config: {args.validation_config}")

        # Log validation setup summary
        if val_cached_text_embeddings:
            logger.info(f"✓ Validation text embeddings: {len(val_cached_text_embeddings)} cached")
        else:
            logger.info("✗ Validation text embeddings: not cached (will be computed on-the-fly)")

        if val_cached_image_embeddings:
            logger.info(f"✓ Validation image embeddings: {len(val_cached_image_embeddings)} cached")
        else:
            logger.info("✗ Validation image embeddings: not cached (will be computed on-the-fly)")

    # Calculate total number of epochs
    total_samples = len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else len(train_dataloader)
    steps_per_epoch = len(train_dataloader)
    total_epochs = max(1, args.max_train_steps // steps_per_epoch)

    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total epochs: {total_epochs}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    # Initialize global step variable
    initial_global_step = 0

    dataset1 = ToyDataset(num_samples=100, input_dim=10)
    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)

    lora_layers_model, optimizer, _, lr_scheduler = accelerator.prepare(
        lora_layers_model, optimizer, dataloader1, lr_scheduler
    )

    # Handle checkpoint resuming
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Find the latest checkpoint in the output directory
            latest_checkpoint = find_latest_checkpoint(args.output_dir)
            if latest_checkpoint:
                logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                initial_global_step = load_checkpoint(flux_transformer, latest_checkpoint, accelerator)
                if initial_global_step > 0:
                    logger.info(f"Resuming training from step {initial_global_step}")
                    # Note: Training will start from the beginning of the dataset
                    # but the LoRA weights will continue from where they left off
                else:
                    logger.warning("Failed to load checkpoint, starting from step 0")
            else:
                logger.info("No checkpoints found, starting training from scratch")
        else:
            # Specific checkpoint path provided
            if os.path.exists(args.resume_from_checkpoint):
                initial_global_step = load_checkpoint(flux_transformer, args.resume_from_checkpoint, accelerator)
                if initial_global_step > 0:
                    logger.info(f"Resuming training from step {initial_global_step}")
                    # Note: Training will start from the beginning of the dataset
                    # but the LoRA weights will continue from where they left off
                else:
                    logger.warning("Failed to load checkpoint, starting from step 0")
            else:
                logger.warning(f"Checkpoint path {args.resume_from_checkpoint} does not exist, starting from scratch")

    # Set global_step to the loaded checkpoint step (or 0 if starting fresh)
    global_step = initial_global_step

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    if initial_global_step > 0:
        logger.info(f"  RESUMING LoRA weights from checkpoint at step {initial_global_step}")
        logger.info(f"  Training will start from the beginning of the dataset")
        logger.info(f"  Target total steps: {args.max_train_steps}")
    else:
        logger.info("  Starting training from scratch")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total epochs: {total_epochs}")
    if validation_dataloader:
        logger.info(f"  Validation enabled with batch size: {args.validation_config.get('train_batch_size', 'N/A')}")
        logger.info(f"  Validation image size: {args.validation_config.get('img_size', 'N/A')}")
        logger.info(f"  Validation data directory: {args.validation_config.get('img_dir', 'N/A')}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)

            # Initialize epoch tracking
    current_epoch = 0
    epoch_loss = 0.0
    epoch_step_count = 0
    train_loss = 0.0  # Initialize train_loss here

    for epoch in range(total_epochs):
        current_epoch = epoch + 1
        epoch_loss = 0.0
        epoch_step_count = 0
        train_loss = 0.0  # Reset train_loss for each epoch

        logger.info(f"Starting epoch {current_epoch}/{total_epochs}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                if args.precompute_text_embeddings:
                    img, prompt_embeds, prompt_embeds_mask = batch
                    prompt_embeds, prompt_embeds_mask = prompt_embeds.to(dtype=weight_dtype).to(accelerator.device), prompt_embeds_mask.to(dtype=torch.int32).to(accelerator.device)
                else:
                    img, prompts = batch
                with torch.no_grad():
                    if not args.precompute_image_embeddings:
                        pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                        pixel_values = pixel_values.unsqueeze(2)

                        pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    else:
                        pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)

                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std

                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                # Concatenate across channels.
                # pack the latents.
                packed_noisy_model_input = QwenImagePipeline._pack_latents(
                    noisy_model_input,
                    bsz,
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                # latent image ids for RoPE.
                img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz
                with torch.no_grad():
                    if not args.precompute_text_embeddings:
                        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                            prompt=prompts,
                            device=packed_noisy_model_input.device,
                            num_images_per_prompt=1,
                            max_sequence_length=1024,
                        )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                # flow-matching loss
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                epoch_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_step_count += 1

                # Get current learning rate
                current_lr = lr_scheduler.get_last_lr()[0]

                # Log comprehensive metrics to W&B
                accelerator.log({
                    "train_loss": train_loss,
                    "learning_rate": current_lr,
                    "epoch_progress": epoch_step_count / steps_per_epoch,
                }, step=global_step)

                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    #accelerator.save_state(save_path)
                    try:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                    except:
                        pass
                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )

                    QwenImagePipeline.save_lora_weights(
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,
                    )

                    logger.info(f"Saved state to {save_path}")

                                        # Perform validation if validation dataset is available
                    if validation_dataloader:
                        logger.info("Computing validation loss...")
                        # Only create text encoding pipeline if we don't have cached embeddings
                        val_text_pipeline = None
                        if not args.precompute_text_embeddings:
                            if 'text_encoding_pipeline' in locals():
                                val_text_pipeline = text_encoding_pipeline
                            else:
                                # Create a new one for validation only if needed
                                val_text_pipeline = QwenImagePipeline.from_pretrained(
                                    args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
                                )
                                val_text_pipeline.to(accelerator.device)

                        logger.info(f"Starting checkpoint validation at step {global_step}...")
                        val_loss = compute_validation_loss(
                            flux_transformer, vae, val_text_pipeline, noise_scheduler_copy,
                            validation_dataloader, accelerator, weight_dtype, get_sigmas
                        )
                        logger.info(f"Checkpoint validation completed at step {global_step}: {val_loss:.6f}")

                        # Log validation loss to W&B
                        accelerator.log({
                            "checkpoint_validation_loss": val_loss,
                            "epoch_progress": epoch_step_count / steps_per_epoch,
                        }, step=global_step)

                        # Clean up temporary text pipeline if we created one
                        if val_text_pipeline is not None and val_text_pipeline != text_encoding_pipeline:
                            del val_text_pipeline
                            torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # Log epoch-level metrics
        if epoch_step_count > 0:
            avg_epoch_loss = epoch_loss / epoch_step_count
            accelerator.log({
                "epoch_loss": avg_epoch_loss,
                "epoch_progress": epoch_step_count / steps_per_epoch,
            }, step=global_step)
            logger.info(f"Epoch {current_epoch} completed. Average loss: {avg_epoch_loss:.4f}")

                        # Perform validation at the end of each epoch if validation dataset is available
            if validation_dataloader:
                logger.info(f"Computing validation loss for epoch {current_epoch}...")
                # Only create text encoding pipeline if we don't have cached embeddings
                val_text_pipeline = None
                if not args.precompute_text_embeddings:
                    if 'text_encoding_pipeline' in locals():
                        val_text_pipeline = text_encoding_pipeline
                    else:
                        # Create a new one for validation only if needed
                        val_text_pipeline = QwenImagePipeline.from_pretrained(
                            args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
                        )
                        val_text_pipeline.to(accelerator.device)

                logger.info(f"Starting epoch {current_epoch} validation...")
                val_loss = compute_validation_loss(
                    flux_transformer, vae, val_text_pipeline, noise_scheduler_copy,
                    validation_dataloader, accelerator, weight_dtype, get_sigmas
                )
                logger.info(f"Epoch {current_epoch} validation completed: {val_loss:.6f}")

                # Log epoch validation loss to W&B
                accelerator.log({
                    "epoch_validation_loss": val_loss,
                    "epoch_progress": epoch_step_count / steps_per_epoch,
                }, step=global_step)

                # Clean up temporary text pipeline if we created one
                if val_text_pipeline is not None and val_text_pipeline != text_encoding_pipeline:
                    del val_text_pipeline
                    torch.cuda.empty_cache()

            # Log additional epoch statistics
            if accelerator.is_main_process:
                logger.info(f"Epoch {current_epoch} Summary:")
                logger.info(f"  - Steps completed: {epoch_step_count}")
                logger.info(f"  - Average loss: {avg_epoch_loss:.6f}")
                if validation_dataloader:
                    logger.info(f"  - Validation loss: {val_loss:.6f}")
                logger.info(f"  - Current learning rate: {lr_scheduler.get_last_lr()[0]:.2e}")
                logger.info(f"  - Global step: {global_step}")
                logger.info(f"  - Progress: {global_step}/{args.max_train_steps} ({100*global_step/args.max_train_steps:.1f}%)")

    # Final logging
    if accelerator.is_main_process:
        logger.info("Training completed!")
        logger.info(f"Total steps: {global_step}")
        logger.info(f"Total epochs: {current_epoch}")
        logger.info(f"Final learning rate: {lr_scheduler.get_last_lr()[0]:.2e}")


    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
