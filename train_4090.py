"""Resumable, single-GPU Qwen-Image LoRA training with persistent checkpoints."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch.utils.data import DataLoader

from utils.training_checkpoint import load_checkpoint, resolve_checkpoint, restore_checkpoint_state, save_checkpoint
from utils.training_data import (
    CachedTrainingDataset, atomic_torch_save, cache_is_valid, dataset_fingerprint,
    list_training_samples, write_cache_signature,
)
from utils.training_runtime import (
    StopRequest, build_run_signature, infinite_batches, train_steps, validate_training_config,
)

LOGGER = logging.getLogger(__name__)
MODEL_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML training recipe")
    parser.add_argument("--dataset-dir", help="Override data_config.img_dir")
    parser.add_argument("--output-dir", help="Override output_dir (persistent disk)")
    parser.add_argument("--max-train-steps", type=int, help="Total optimizer-step target; keep unchanged when resuming")
    parser.add_argument("--stop-after-steps", type=int, help="Gracefully stop after this many optimizer steps in this session")
    parser.add_argument("--resume-from-checkpoint", help="latest, checkpoint path, or none")
    parser.add_argument("--preflight", action="store_true", help="Validate config/dataset/checkpoint selection without CUDA or downloads")
    return parser.parse_args(argv)


def load_config(cli):
    config = OmegaConf.to_container(OmegaConf.load(cli.config), resolve=True)
    config.setdefault("model_revision", MODEL_REVISION)
    config.setdefault("seed", 42)
    config.setdefault("max_sequence_length", 1024)
    if cli.dataset_dir:
        config["data_config"]["img_dir"] = cli.dataset_dir
    if cli.output_dir:
        config["output_dir"] = cli.output_dir
    if cli.max_train_steps is not None:
        config["max_train_steps"] = cli.max_train_steps
    if cli.resume_from_checkpoint is not None:
        config["resume_from_checkpoint"] = cli.resume_from_checkpoint
    if config.get("resume_from_checkpoint") == "none":
        config["resume_from_checkpoint"] = None
    if cli.stop_after_steps is not None and cli.stop_after_steps <= 0:
        raise ValueError("--stop-after-steps must be positive")
    validate_training_config(config)
    return config


def runtime_versions():
    versions = {}
    for name in ("torch", "diffusers", "peft", "transformers", "optimum-quanto",
                 "bitsandbytes", "numpy", "Pillow", "huggingface-hub"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    direct_url = metadata.distribution("diffusers").read_text("direct_url.json")
    if direct_url:
        versions["diffusers_commit"] = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    return versions


@contextmanager
def output_lock(output_dir):
    import fcntl
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".training.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another trainer is using {output_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _resize(image, maximum):
    scale = maximum / max(image.size)
    width, height = (int(side * scale) // 32 * 32 for side in image.size)
    if min(width, height) < 32:
        raise ValueError("Image aspect ratio is too extreme for this img_size")
    return image.resize((width, height))


def prepare_caches(config, samples, data_hash, device, dtype, stop):
    """Reuse completed files after interruption; publish each file atomically."""
    from diffusers import AutoencoderKLQwenImage, QwenImagePipeline
    recipe = {
        "model": config["pretrained_model_name_or_path"], "revision": config["model_revision"],
        "dataset": data_hash, "seed": config["seed"], "img_size": config["data_config"]["img_size"],
        "max_sequence_length": config["max_sequence_length"], "dtype": str(dtype),
        "versions": runtime_versions(), "format": 1,
    }
    namespace = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(config["output_dir"]) / "cache" / namespace
    text_dir, image_dir = root / "text", root / "image"
    required_text = [sample.caption_cache_name for sample in samples] + ["empty_embedding.pt"]
    required_image = [sample.image_cache_name for sample in samples]
    model_kwargs = {"revision": config["model_revision"], "torch_dtype": dtype}
    stop.raise_if_requested()
    if not cache_is_valid(text_dir, recipe, required_text):
        write_cache_signature(text_dir, recipe)
        pipeline = QwenImagePipeline.from_pretrained(
            config["pretrained_model_name_or_path"], transformer=None, vae=None, **model_kwargs,
        ).to(device)
        captions = [(sample.caption_cache_name, sample.caption_path.read_text(encoding="utf-8")) for sample in samples]
        for name, prompt in captions + [("empty_embedding.pt", " ")]:
            stop.raise_if_requested()
            if cache_is_valid(text_dir, recipe, [name]):
                continue
            with torch.no_grad():
                embedding, mask = pipeline.encode_prompt(
                    prompt=[prompt], device=device, num_images_per_prompt=1,
                    max_sequence_length=config["max_sequence_length"],
                )
            atomic_torch_save({"prompt_embeds": embedding[0].cpu(), "prompt_embeds_mask": mask[0].cpu()}, text_dir / name)
            stop.raise_if_requested()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    vae_config = AutoencoderKLQwenImage.load_config(
        config["pretrained_model_name_or_path"], subfolder="vae", revision=config["model_revision"],
    )
    if not cache_is_valid(image_dir, recipe, required_image):
        write_cache_signature(image_dir, recipe)
        vae = AutoencoderKLQwenImage.from_pretrained(
            config["pretrained_model_name_or_path"], subfolder="vae", **model_kwargs,
        ).to(device)
        for sample in samples:
            stop.raise_if_requested()
            if cache_is_valid(image_dir, recipe, [sample.image_cache_name]):
                continue
            with Image.open(sample.image_path) as image:
                image = _resize(image.convert("RGB"), config["data_config"]["img_size"])
                pixels = torch.from_numpy(np.array(image).astype(np.float32) / 127.5 - 1)
            pixels = pixels.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
            image_seed = int(hashlib.sha256(f"{config['seed']}:{sample.image_path.name}".encode()).hexdigest()[:16], 16)
            generator = torch.Generator(device=device).manual_seed(image_seed)
            with torch.no_grad():
                latent = vae.encode(pixels).latent_dist.sample(generator=generator)[0].cpu()
            atomic_torch_save(latent, image_dir / sample.image_cache_name)
            stop.raise_if_requested()
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    if not cache_is_valid(text_dir, recipe, required_text) or not cache_is_valid(image_dir, recipe, required_image):
        raise ValueError("Embedding cache did not finish correctly")
    vae_config = dict(vae_config)
    vae_config["vae_scale_factor"] = 2 ** len(vae_config["temperal_downsample"])
    return image_dir, text_dir, vae_config


def qwen_flow_loss(model, batch, scheduler, vae_config, device, dtype):
    """Original Qwen flow-matching objective, exercised with a tiny CPU model."""
    from diffusers import QwenImagePipeline
    from diffusers.training_utils import compute_density_for_timestep_sampling
    latent, embedding, mask = batch
    latent = latent.to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)
    channels = vae_config["z_dim"]
    mean = torch.tensor(vae_config["latents_mean"], device=device, dtype=dtype).view(1, 1, channels, 1, 1)
    std = torch.tensor(vae_config["latents_std"], device=device, dtype=dtype).view(1, 1, channels, 1, 1)
    latent = (latent - mean) / std
    batch_size = latent.shape[0]
    noise = torch.randn_like(latent)
    density = compute_density_for_timestep_sampling(
        weighting_scheme="none", batch_size=batch_size, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
    )
    indices = (density * scheduler.config.num_train_timesteps).long()
    timesteps = scheduler.timesteps[indices].to(device)
    sigma = scheduler.sigmas[indices].to(device=device, dtype=dtype).view(batch_size, 1, 1, 1, 1)
    noisy = (1 - sigma) * latent + sigma * noise
    packed = QwenImagePipeline._pack_latents(noisy, batch_size, channels, noisy.shape[3], noisy.shape[4])
    embedding = embedding.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=torch.int32)
    predicted = model(
        hidden_states=packed, timestep=timesteps / 1000, guidance=None,
        encoder_hidden_states_mask=mask, encoder_hidden_states=embedding,
        img_shapes=[(1, noisy.shape[3] // 2, noisy.shape[4] // 2)] * batch_size,
        txt_seq_lens=mask.sum(dim=1).tolist(), return_dict=False,
    )[0]
    scale = vae_config["vae_scale_factor"]
    predicted = QwenImagePipeline._unpack_latents(predicted, noisy.shape[3] * scale, noisy.shape[4] * scale, scale)
    target = (noise - latent).permute(0, 2, 1, 3, 4)
    return (predicted.float() - target.float()).square().mean()


def ensure_inference_export(checkpoint_path, adapters):
    """Regenerate derived inference weights atomically, including after a killed export."""
    from diffusers import QwenImagePipeline
    checkpoint_path = Path(checkpoint_path)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=checkpoint_path.parent) as temporary:
        QwenImagePipeline.save_lora_weights(temporary, transformer_lora_layers=adapters)
        source = Path(temporary) / "pytorch_lora_weights.safetensors"
        with source.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(source, checkpoint_path / source.name)


def extract_adapter_state(model):
    # PEFT's default first calls model.state_dict(), unpacking the entire FP8 base.
    # Passing only trainable parameters avoids both that allocation and base serialization.
    from peft.utils import get_peft_model_state_dict
    return get_peft_model_state_dict(model, state_dict={
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }, save_embedding_layers=False)


def quantize_qwen_transformer(model, device):
    from optimum.quanto import freeze, qfloat8, quantize
    from tqdm.auto import tqdm
    for block in tqdm(model.transformer_blocks, desc="Quantizing transformer"):
        block.to(device)
        quantize(block, weights=qfloat8)
        freeze(block)
        block.to("cpu")
    # CUDA autocast runs norm_out in FP32. Native Linear casts that output to
    # BF16; Quanto's Marlin kernel bypasses autocast and rejects FP32 inputs.
    quantize(model, weights=qfloat8, exclude=["proj_out"])
    freeze(model)


def train(config, samples, data_hash, signature, resume_path, stop_after_steps, stop):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from tqdm.auto import tqdm
    output_dir = Path(config["output_dir"])
    payload = load_checkpoint(resume_path, signature) if resume_path else None
    global_step = payload["global_step"] if payload else 0
    if global_step == config["max_train_steps"]:
        ensure_inference_export(resume_path, payload["adapter_state"])
        LOGGER.info("Training already complete at step %s: %s", global_step, resume_path)
        return 0
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU required for training; use --preflight or CPU tests on Mac")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU, for example with CUDA_VISIBLE_DEVICES, for this recipe")
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This recipe requires a BF16-capable CUDA GPU")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    image_dir, text_dir, vae_config = prepare_caches(config, samples, data_hash, device, dtype, stop)
    stop.raise_if_requested()
    model = QwenImageTransformer2DModel.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="transformer",
        revision=config["model_revision"], torch_dtype=dtype,
    )
    model.requires_grad_(False)
    if config["quantize"]:
        quantize_qwen_transformer(model, device)
    model.requires_grad_(False)
    model.to(device)
    # Cache hits/misses and model construction must not change fresh LoRA initialization.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model.add_adapter(LoraConfig(
        r=config["rank"], lora_alpha=config["rank"], init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    stop.raise_if_requested()
    model.enable_gradient_checkpointing()
    if config["adam8bit"]:
        import bitsandbytes as bnb
        optimizer_type = bnb.optim.AdamW8bit
    else:
        optimizer_type = torch.optim.AdamW
    optimizer = optimizer_type(parameters, lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["adam_weight_decay"], eps=config["adam_epsilon"])
    scheduler = get_scheduler(config["lr_scheduler"], optimizer,
        num_warmup_steps=config["lr_warmup_steps"], num_training_steps=config["max_train_steps"])
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="scheduler", revision=config["model_revision"],
    )
    dataset = CachedTrainingDataset(samples, image_dir, text_dir, config["data_config"]["caption_dropout_rate"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"]))
    batches = infinite_batches(loader)
    if payload:
        if set(payload["adapter_state"]) != set(extract_adapter_state(model)):
            raise ValueError("Checkpoint adapter keys do not match this model")
        loaded = set_peft_model_state_dict(model, payload["adapter_state"])
        if loaded.unexpected_keys:
            raise ValueError(f"Unexpected adapter keys: {loaded.unexpected_keys}")
        restore_checkpoint_state(payload, optimizer, scheduler)
        LOGGER.info("Resumed %s at optimizer step %s", resume_path, global_step)
    (output_dir / "run_manifest.json").write_text(json.dumps({
        "signature": signature, "gpu": torch.cuda.get_device_name(device),
        "cuda": torch.version.cuda, "resumed_from": str(resume_path) if resume_path else None,
    }, indent=2) + "\n", encoding="utf-8")
    progress = tqdm(total=config["max_train_steps"], initial=global_step, desc="Optimizer steps")

    def checkpoint(step):
        adapters = extract_adapter_state(model)
        path = save_checkpoint(output_dir, step, adapters, optimizer, scheduler, signature,
            checkpoints_total_limit=config.get("checkpoints_total_limit", 3))
        # Export is derived; a killed export does not invalidate the full checkpoint.
        ensure_inference_export(path, adapters)
        LOGGER.info("Saved complete checkpoint and inference weights: %s", path)

    def on_step(step, loss):
        progress.update(1)
        progress.set_postfix(loss=loss)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "loss": loss, "lr": scheduler.get_last_lr()[0],
                "session_start_step": global_step,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device)}) + "\n")

    try:
        result = train_steps(model, optimizer, scheduler, batches,
            lambda batch: qwen_flow_loss(model, batch, noise_scheduler, vae_config, device, dtype),
            global_step=global_step, max_train_steps=config["max_train_steps"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            checkpointing_steps=config["checkpointing_steps"], save_checkpoint=checkpoint,
            stop_after_steps=stop_after_steps, stop_request=stop,
            max_grad_norm=config["max_grad_norm"],
            autocast=lambda: torch.autocast("cuda", dtype=dtype), on_step=on_step)
    finally:
        progress.close()
    LOGGER.info("Stopped at optimizer step %s (%s)", result.global_step, result.reason)
    return 128 + stop.signum if result.reason == "signal" else 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli = parse_args(argv)
    config = load_config(cli)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The resumable 4090 entry point supports exactly one GPU/process")
    samples = list_training_samples(config["data_config"]["img_dir"], config["data_config"].get("caption_type", "txt"))
    data_hash = dataset_fingerprint(samples)
    output_dir = Path(config["output_dir"])
    if cli.preflight:
        resume_path = resolve_checkpoint(output_dir, config.get("resume_from_checkpoint"))
        print(json.dumps({"samples": len(samples), "dataset_sha256": data_hash,
            "model_revision": config["model_revision"], "output_dir": str(output_dir),
            "resume_checkpoint": str(resume_path) if resume_path else None}, indent=2))
        return 0
    signature = build_run_signature(config, data_hash, runtime_versions())
    with output_lock(output_dir), StopRequest() as stop:
        resume_path = resolve_checkpoint(output_dir, config.get("resume_from_checkpoint"))
        if resume_path is None and any(output_dir.glob("checkpoint-*")):
            raise ValueError("Use a new output_dir for a fresh run; existing checkpoints must not be overwritten")
        return train(config, samples, data_hash, signature, resume_path, cli.stop_after_steps, stop)


if __name__ == "__main__":
    raise SystemExit(main())
