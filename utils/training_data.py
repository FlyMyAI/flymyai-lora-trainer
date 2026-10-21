"""Validated image/caption inputs and disk caches for resumable training."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrainingSample:
    image_path: Path
    caption_path: Path

    @property
    def image_cache_name(self):
        return self.image_path.name + ".pt"

    @property
    def caption_cache_name(self):
        return self.caption_path.name + ".pt"


def list_training_samples(img_dir, caption_type="txt"):
    """Validate all pairs before model downloads or paid GPU computation.

    Stems and supported suffixes are compared without case. Shared stems are
    refused because they would make captions and cached embeddings ambiguous.
    """
    directory = Path(img_dir)
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist: {directory}")
    caption_suffix = "." + caption_type.lstrip(".").lower()
    paths = sorted((path for path in directory.iterdir() if path.is_file()), key=lambda p: (p.name.casefold(), p.name))
    images = [path for path in paths if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        raise ValueError(f"No supported training images in {directory}")
    captions = {}
    for path in paths:
        if path.suffix.lower() == caption_suffix:
            key = path.stem.casefold()
            if key in captions:
                raise ValueError(f"Ambiguous duplicate captions: {captions[key].name}, {path.name}")
            captions[key] = path
    samples = []
    used_stems = {}
    for image_path in images:
        key = image_path.stem.casefold()
        if key in used_stems:
            raise ValueError(f"Ambiguous duplicate image stems: {used_stems[key].name}, {image_path.name}")
        used_stems[key] = image_path
        caption_path = captions.get(key)
        if caption_path is None:
            raise ValueError(f"Missing {caption_suffix} caption for {image_path.name}")
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.load()
                if min(image.size) < 32:
                    raise ValueError("image dimensions must both be at least 32 pixels")
        except Exception as error:
            raise ValueError(f"Invalid training image {image_path}: {error}") from error
        try:
            caption_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"Cannot read UTF-8 caption {caption_path}: {error}") from error
        samples.append(TrainingSample(image_path=image_path, caption_path=caption_path))
    return samples


def dataset_fingerprint(samples):
    """Hash names and actual bytes; moving a dataset leaves its identity intact."""
    digest = hashlib.sha256(b"flymyai-training-data-v1\0")
    for sample in sorted(samples, key=lambda item: (item.image_path.name.casefold(), item.image_path.name)):
        for path in (sample.image_path, sample.caption_path):
            name = path.name.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _is_embedding_payload(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("prompt_embeds"), torch.Tensor)
        and isinstance(value.get("prompt_embeds_mask"), torch.Tensor)
        and value["prompt_embeds"].numel() > 0
        and value["prompt_embeds_mask"].numel() > 0
    )


def _load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Cannot read cache {path}; rebuild the embedding cache: {error}") from error


class CachedTrainingDataset(Dataset):
    """Sample cached pairs with replacement using the restorable Python RNG.

    Use a single process and ``num_workers=0`` so checkpointed RNG state also
    covers sample selection and caption dropout. Bad samples fail immediately.
    """

    def __init__(self, samples, image_cache_dir, text_cache_dir, caption_dropout_rate=0):
        self.samples = list(samples)
        self.image_cache_dir = Path(image_cache_dir)
        self.text_cache_dir = Path(text_cache_dir)
        self.caption_dropout_rate = float(caption_dropout_rate)
        if not self.samples:
            raise ValueError("Cannot train on an empty dataset")
        if not math.isfinite(self.caption_dropout_rate) or not 0 <= self.caption_dropout_rate <= 1:
            raise ValueError("caption_dropout_rate must be between 0 and 1")
        required = [self.image_cache_dir / sample.image_cache_name for sample in self.samples]
        required += [self.text_cache_dir / sample.caption_cache_name for sample in self.samples]
        if self.caption_dropout_rate > 0:
            required.append(self.text_cache_dir / "empty_embedding.pt")
        for path in required:
            if not path.is_file():
                raise ValueError(f"Missing training cache {path}; rebuild the embedding cache")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = random.choice(self.samples)
        image_path = self.image_cache_dir / sample.image_cache_name
        latent = _load_cache(image_path)
        if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
            raise ValueError(f"Invalid latent tensor in {image_path}; rebuild the embedding cache")
        caption_name = sample.caption_cache_name
        if self.caption_dropout_rate > 0 and random.random() < self.caption_dropout_rate:
            caption_name = "empty_embedding.pt"
        text_path = self.text_cache_dir / caption_name
        embedding = _load_cache(text_path)
        if not _is_embedding_payload(embedding):
            raise ValueError(f"Invalid text embedding in {text_path}; rebuild the embedding cache")
        return latent, embedding["prompt_embeds"], embedding["prompt_embeds_mask"]


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".cache-", delete=False) as stream:
            temporary_path = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_torch_save(value, path):
    """Publish only a completely written tensor cache, on the same filesystem."""
    _atomic_write(path, lambda stream: torch.save(value, stream))


def write_cache_signature(cache_dir, signature):
    payload = json.dumps({"signature": signature}, sort_keys=True).encode("utf-8")
    _atomic_write(Path(cache_dir) / ".signature.json", lambda stream: stream.write(payload))


def cache_is_valid(cache_dir, signature, required_files):
    """Reuse only a matching recipe with every required cache readable on CPU."""
    cache_dir = Path(cache_dir)
    try:
        metadata = json.loads((cache_dir / ".signature.json").read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return False
        for name in required_files:
            value = _load_cache(cache_dir / name)
            if not ((isinstance(value, torch.Tensor) and value.numel() > 0) or _is_embedding_payload(value)):
                return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True
