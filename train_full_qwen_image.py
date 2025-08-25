"""
Full Training Script for Qwen Image Model
Performs complete fine-tuning of all model parameters instead of LoRA adaptation
"""

import argparse
import copy
from copy import deepcopy
import math
import os
import shutil

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
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.dataset import loader
from omegaconf import OmegaConf
import transformers
from loguru import logger as loguru_logger
import bitsandbytes as bnb
import gc

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Full training script for Qwen Image model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to training config file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    return args.config, args.resume_from_checkpoint


def calculate_model_size(model):
    """Calculate model size in millions of parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    loguru_logger.info(f"Total parameters: {total_params / 1_000_000:.2f}M")
    loguru_logger.info(f"Trainable parameters: {trainable_params / 1_000_000:.2f}M")
    loguru_logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return total_params, trainable_params


def setup_model_for_training(model, freeze_text_encoder=True, freeze_vae=True):
    """Setup model parameters for full training"""
    
    # Enable training mode
    model.train()
    
    # Enable gradient computation for transformer
    model.requires_grad_(True)
    
    # Optional: freeze certain components to save memory
    if hasattr(model, 'text_encoder') and freeze_text_encoder:
        model.text_encoder.requires_grad_(False)
        loguru_logger.info("Text encoder frozen")
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        loguru_logger.info("Gradient checkpointing enabled")
    
    return model


def save_full_model(model, save_path, accelerator, args):
    """Save the full model state"""
    loguru_logger.info(f"Saving full model to {save_path}")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Unwrap model from accelerator
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model._orig_mod if is_compiled_module(unwrapped_model) else unwrapped_model
    
    # Save transformer
    transformer_path = os.path.join(save_path, "transformer")
    unwrapped_model.save_pretrained(transformer_path, safe_serialization=True)
    
    # Save config
    config_path = os.path.join(save_path, "config.yaml")
    OmegaConf.save(args, config_path)
    
    loguru_logger.info(f"Model saved successfully to {save_path}")


def main():
    config_path, resume_checkpoint = parse_args()
    args = OmegaConf.load(config_path)
    
    # Setup logging directory
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # Setup loguru logger
    loguru_logger.add(
        os.path.join(logging_dir, "training.log"),
        rotation="100 MB",
        retention="10 days",
        level="INFO"
    )
    
    # Setup basic logging
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    loguru_logger.info(f"Using weight dtype: {weight_dtype}")
    
    # Load models
    loguru_logger.info("Loading models...")
    
    # Text encoding pipeline (for prompt encoding)
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        transformer=None, 
        vae=None, 
        torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)
    
    # VAE
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)  # VAE is typically frozen during training
    
    # Main transformer model
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    
    # Setup model for full training
    flux_transformer = setup_model_for_training(
        flux_transformer, 
        freeze_text_encoder=getattr(args, 'freeze_text_encoder', True),
        freeze_vae=True
    )
    
    # Move to device
    flux_transformer.to(accelerator.device, dtype=weight_dtype)
    
    # Calculate and log model size
    calculate_model_size(flux_transformer)
    
    # Noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """Get noise sigmas for given timesteps"""
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # Setup optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    
    # Use AdamW or Adam8bit based on config
    if getattr(args, 'use_8bit_adam', False):
        loguru_logger.info("Using 8-bit Adam optimizer")
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        loguru_logger.info("Using standard AdamW optimizer")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    # Setup data loader
    loguru_logger.info("Setting up data loader...")
    train_dataloader = loader(**args.data_config)
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with accelerator
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Resume from checkpoint if specified
    global_step = 0
    initial_global_step = 0
    
    if resume_checkpoint:
        if resume_checkpoint == "latest":
            # Find latest checkpoint
            dirs = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                resume_checkpoint = dirs[-1]
                resume_checkpoint = os.path.join(args.output_dir, resume_checkpoint)
        
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            loguru_logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            accelerator.load_state(resume_checkpoint)
            global_step = int(resume_checkpoint.split("-")[-1])
            initial_global_step = global_step
    
    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"config": OmegaConf.to_yaml(args)})
    
    # Calculate total batch size
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    # Training info
    loguru_logger.info("***** Running training *****")
    loguru_logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    loguru_logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    loguru_logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    loguru_logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    loguru_logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # VAE scale factor
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    
    # Training loop
    for epoch in range(args.num_train_epochs if hasattr(args, 'num_train_epochs') else 1):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                img, prompts = batch
                
                with torch.no_grad():
                    # Encode images to latents
                    pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                    pixel_values = pixel_values.unsqueeze(2)
                    
                    pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                    
                    # Normalize latents
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std
                    
                    # Sample noise and timesteps
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
                
                # Get sigmas and create noisy input
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                
                # Pack latents
                packed_noisy_model_input = QwenImagePipeline._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                
                # Image shapes for RoPE
                img_shapes = [(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2)] * bsz
                
                with torch.no_grad():
                    # Encode prompts
                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                        prompt=prompts,
                        device=packed_noisy_model_input.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                
                # Forward pass through transformer
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
                
                # Unpack predictions
                model_pred = QwenImagePipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Calculate loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Gather losses for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # Gradient clipping
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Check if we performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                accelerator.log({
                    "train_loss": train_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0]
                }, step=global_step)
                train_loss = 0.0
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Manage checkpoint limit
                        if hasattr(args, 'checkpoints_total_limit') and args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]
                                
                                loguru_logger.info(
                                    f"Removing {len(removing_checkpoints)} old checkpoints"
                                )
                                
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        # Save new checkpoint
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_full_model(flux_transformer, save_path, accelerator, args)
            
            # Update progress bar
            logs = {
                "step_loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            
            # Check if we've reached max training steps
            if global_step >= args.max_train_steps:
                break
        
        # Break epoch loop if we've reached max steps
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model")
        save_full_model(flux_transformer, final_save_path, accelerator, args)
        loguru_logger.info("Training completed successfully!")
    
    # Wait for all processes and end training
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
