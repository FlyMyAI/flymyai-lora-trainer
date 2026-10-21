"""Single-process optimizer loop with checkpoints only at complete update boundaries."""
from contextlib import nullcontext
from dataclasses import dataclass
from copy import deepcopy
import re
import signal

import torch


def validate_training_config(config):
    """Bound this entry point to the single-GPU, cached BF16 recipe we can resume."""
    data = config["data_config"]
    if data.get("num_workers", 0) != 0:
        raise ValueError("num_workers must be 0 to restore sample/dropout RNG exactly")
    if config.get("mixed_precision") != "bf16":
        raise ValueError("The resumable 4090 trainer supports mixed_precision: bf16")
    for option in ("precompute_text_embeddings", "precompute_image_embeddings", "save_cache_on_disk"):
        if config.get(option) is not True:
            raise ValueError(f"{option} must be true for the resumable 4090 recipe")
    if data.get("train_batch_size") != 1 or config.get("train_batch_size") != 1:
        raise ValueError("The resumable 4090 recipe requires train_batch_size: 1")
    if data.get("random_ratio", False):
        raise ValueError("random_ratio must be false when using fixed latent caches")
    if data["img_size"] < 32 or data["img_size"] % 32:
        raise ValueError("img_size must be a positive multiple of 32")
    for option in ("max_train_steps", "checkpointing_steps", "gradient_accumulation_steps", "rank"):
        if isinstance(config[option], bool) or not isinstance(config[option], int) or config[option] <= 0:
            raise ValueError(f"{option} must be a positive integer")
    limit = config.get("checkpoints_total_limit", 3)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("checkpoints_total_limit must be positive or null")
    if not re.fullmatch(r"[0-9a-f]{40}", str(config.get("model_revision", ""))):
        raise ValueError("model_revision must be an immutable Hugging Face commit SHA")
    if config.get("pretrained_model_name_or_path") != "Qwen/Qwen-Image":
        raise ValueError("This entry point currently supports Qwen/Qwen-Image only")
    if config.get("report_to") is not None:
        raise ValueError("report_to must be null; this recipe writes local metrics.jsonl")


def build_run_signature(config, dataset_hash, versions):
    training = deepcopy(config)
    for key in ("output_dir", "logging_dir", "tracker_project_name", "resume_from_checkpoint",
                "checkpointing_steps", "checkpoints_total_limit", "cache_dir"):
        training.pop(key, None)
    training["data_config"].pop("img_dir", None)
    return {"trainer_format": 1, "training": training, "dataset_sha256": dataset_hash, "versions": versions}


@dataclass(frozen=True)
class TrainingResult:
    global_step: int
    reason: str


class StopRequest:
    """Signal handlers only set a flag; serialization runs in the training loop."""

    def __init__(self):
        self.signum = None
        self._previous = {}

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._request)
        return self

    def _request(self, signum, _frame):
        self.signum = signum

    def raise_if_requested(self):
        """Honor stops between durable cache writes, before training has started."""
        if self.signum is not None:
            raise SystemExit(128 + self.signum)

    def __exit__(self, *_exc):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def infinite_batches(loader):
    """Recreate iterators without retaining batches (unlike itertools.cycle)."""
    while True:
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise ValueError("Training dataset is empty")


def train_steps(
    model, optimizer, scheduler, batches, loss_fn, *, max_train_steps,
    checkpointing_steps, save_checkpoint, global_step=0,
    gradient_accumulation_steps=1, stop_after_steps=None, stop_request=None,
    max_grad_norm=None, autocast=nullcontext, on_step=None,
):
    if max_train_steps <= 0 or checkpointing_steps <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Step counts and gradient accumulation must be positive")
    if global_step < 0 or global_step > max_train_steps:
        raise ValueError("Checkpoint step exceeds the total training target")
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError("stop_after_steps must be positive")
    if global_step == max_train_steps:
        return TrainingResult(global_step, "complete")

    session_start = global_step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    while global_step < max_train_steps:
        loss_sum = 0.0
        for _ in range(gradient_accumulation_steps):
            with autocast():
                loss = loss_fn(next(batches))
            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError("Non-finite training loss; previous checkpoint retained")
            loss_sum += loss.detach().float().item() / gradient_accumulation_steps
            (loss / gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, float("inf") if max_grad_norm is None else max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if on_step is not None:
            on_step(global_step, loss_sum)

        reason = None
        if stop_request is not None and stop_request.signum is not None:
            reason = "signal"
        elif global_step >= max_train_steps:
            reason = "complete"
        elif stop_after_steps is not None and global_step - session_start >= stop_after_steps:
            reason = "session_limit"
        if global_step % checkpointing_steps == 0 or reason is not None:
            save_checkpoint(global_step)
        if reason is not None:
            return TrainingResult(global_step, reason)
