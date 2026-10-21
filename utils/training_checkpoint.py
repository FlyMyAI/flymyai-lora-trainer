"""Atomic adapter-only checkpoints for trusted, single-process training runs.

Load adapters first, then call ``restore_checkpoint_state`` immediately before
continuing the training loop. Training state uses pickle and must only be loaded
from checkpoints produced by a trusted local training run.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import uuid

import numpy as np
import torch
from safetensors.torch import load_file, save_file


_FORMAT_VERSION = 1
_CHECKPOINT_NAME = re.compile(r"checkpoint-(\d+)$")
_PAYLOAD_FILES = ("adapter.safetensors", "training_state.pt", "metadata.json")


class CheckpointError(ValueError):
    """A checkpoint is incomplete, corrupt, legacy, or incompatible with a run."""


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False, indent=2)
        handle.write("\n")


def _fsync_directory(path):
    # Directory fsync is unavailable on some filesystems; payload fsync is mandatory.
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _validate_checkpoint(path):
    path = Path(path)
    try:
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if not match or path.is_symlink() or not path.is_dir():
            raise ValueError("expected a complete checkpoint-<step> directory")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("missing manifest; legacy adapter-only checkpoints cannot resume training")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["format_version"] != _FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format")
        if set(manifest["files"]) != set(_PAYLOAD_FILES):
            raise ValueError("incomplete checkpoint payload")
        for name in _PAYLOAD_FILES:
            payload_path = path / name
            if payload_path.is_symlink() or not payload_path.is_file():
                raise ValueError(f"missing payload: {name}")
            if _file_digest(payload_path) != manifest["files"][name]:
                raise ValueError(f"integrity check failed: {name}")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata["format_version"] != _FORMAT_VERSION or metadata["global_step"] != int(match[1]):
            raise ValueError("checkpoint step or format does not match metadata")
        if not isinstance(metadata["signature"], dict):
            raise ValueError("checkpoint signature is missing")
        return metadata
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CheckpointError(f"Invalid checkpoint {path}: {exc}") from exc


def _complete_checkpoints(output_dir):
    complete = []
    for path in Path(output_dir).iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match:
            try:
                _validate_checkpoint(path)
            except CheckpointError:
                continue
            complete.append((int(match[1]), path))
    return sorted(complete)


def resolve_checkpoint(output_dir, resume_from_checkpoint="latest"):
    """Resolve a full checkpoint; an empty ``latest`` search starts a new run.

    Existing legacy or corrupt checkpoints never silently become a fresh run.
    ``latest`` skips incomplete directories and checkpoints failing their hashes.
    """
    output_dir = Path(output_dir)
    if resume_from_checkpoint is None:
        return None
    if str(resume_from_checkpoint) != "latest":
        path = Path(resume_from_checkpoint)
        if not path.is_absolute() and not path.exists():
            path = output_dir / path
        _validate_checkpoint(path)
        return path
    if not output_dir.exists():
        return None
    complete = _complete_checkpoints(output_dir)
    if complete:
        return complete[-1][1]
    if any(_CHECKPOINT_NAME.fullmatch(path.name) for path in output_dir.iterdir()):
        raise CheckpointError(
            f"No complete valid checkpoint in {output_dir}; legacy adapter-only or corrupt checkpoints cannot resume training"
        )
    return None


def save_checkpoint(
    output_dir,
    step,
    adapter_state,
    optimizer,
    scheduler,
    run_signature,
    checkpoints_total_limit=3,
    scaler=None,
):
    """Publish a complete checkpoint atomically, then prune older valid ones.

    Call only at an optimizer boundary, after gradients have been cleared.
    Frozen model weights are never inspected or serialized.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if checkpoints_total_limit is not None and (
        isinstance(checkpoints_total_limit, bool)
        or not isinstance(checkpoints_total_limit, int)
        or checkpoints_total_limit < 1
    ):
        raise ValueError("checkpoints_total_limit must be a positive integer or None")
    signature = _json_value(run_signature)
    if not isinstance(signature, dict):
        raise ValueError("run_signature must be a JSON object")
    if not adapter_state:
        raise ValueError("adapter_state must contain trainable adapter tensors")
    if any(parameter.grad is not None for group in optimizer.param_groups for parameter in group["params"]):
        raise ValueError("Save only at an optimizer boundary after zero_grad(set_to_none=True); pending gradients cannot resume")

    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"checkpoint-{step}"
    if destination.exists() or destination.is_symlink():
        try:
            _validate_checkpoint(destination)
        except CheckpointError:
            pass
        else:
            raise FileExistsError(f"Checkpoint already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=output_dir))
    try:
        # Clones also detach shared storage that safetensors deliberately rejects.
        save_file(
            {name: value.detach().cpu().contiguous().clone() for name, value in adapter_state.items()},
            temporary / "adapter.safetensors",
        )
        torch.save(training_state, temporary / "training_state.pt")
        _write_json(
            temporary / "metadata.json",
            {
                "format_version": _FORMAT_VERSION,
                "global_step": step,
                "signature": signature,
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
            },
        )
        _write_json(
            temporary / "manifest.json",
            {
                "format_version": _FORMAT_VERSION,
                "files": {name: _file_digest(temporary / name) for name in _PAYLOAD_FILES},
            },
        )
        for name in (*_PAYLOAD_FILES, "manifest.json"):
            with (temporary / name).open("rb") as handle:
                os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if destination.exists() or destination.is_symlink():
            try:
                _validate_checkpoint(destination)
            except CheckpointError:
                # Preserve failed-run evidence only after the replacement is durable.
                # The caller holds the run's single-writer lock throughout training.
                quarantine = output_dir / f".corrupt-{destination.name}-{uuid.uuid4().hex}"
                destination.rename(quarantine)
                _fsync_directory(output_dir)
            else:
                raise FileExistsError(f"Checkpoint already exists: {destination}")
        temporary.rename(destination)
        _fsync_directory(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    if checkpoints_total_limit is not None:
        complete = _complete_checkpoints(output_dir)
        for _, old_path in complete[:-checkpoints_total_limit]:
            shutil.rmtree(old_path)
        _fsync_directory(output_dir)
    return destination


def load_checkpoint(path, expected_signature):
    """Read a trusted checkpoint without changing model, optimizer, or RNG state."""
    path = Path(path)
    metadata = _validate_checkpoint(path)
    if metadata["signature"] != _json_value(expected_signature):
        raise CheckpointError("Checkpoint signature differs from the current model, dataset, seed or training configuration")
    try:
        adapters = load_file(path / "adapter.safetensors", device="cpu")
        # Optimizer and NumPy RNG states are trusted local pickle payloads.
        training_state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
        if not adapters or not {"optimizer", "scheduler", "scaler", "rng"}.issubset(training_state):
            raise ValueError("missing adapter or training state")
        return {
            "adapter_state": adapters,
            "global_step": metadata["global_step"],
            "signature": metadata["signature"],
            "training_state": training_state,
        }
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise CheckpointError(f"Cannot read checkpoint {path}: {exc}") from exc


def restore_checkpoint_state(payload, optimizer, scheduler, scaler=None):
    """Restore training state after the caller loads adapters; restore RNG last."""
    state = payload["training_state"]
    if (state["scaler"] is None) != (scaler is None):
        raise CheckpointError("Checkpoint gradient scaler does not match this run")
    if (state["scheduler"] is None) != (scheduler is None):
        raise CheckpointError("Checkpoint scheduler does not match this run")
    if state["rng"]["cuda"] is not None and not torch.cuda.is_available():
        raise CheckpointError("Checkpoint requires CUDA RNG state but CUDA is unavailable")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
