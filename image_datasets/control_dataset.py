import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1, cached_text_embeddings=None,
                 cached_image_embeddings=None, control_dir=None, cached_image_embeddings_control=None):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.control_dir = control_dir
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control
        print('cached_text_embeddings', type(cached_text_embeddings))
    def __len__(self):
        return 999999

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                idx = random.randint(0, len(self.images) - 1)
                img_path = self.images[idx]
                img_name = os.path.basename(img_path)
                base = os.path.splitext(img_name)[0]
                txt = base + ".txt"

                # --- image load (cached or not) ---
                if self.cached_image_embeddings is None:
                    img = Image.open(img_path).convert("RGB")
                    img = image_resize(img, self.img_size)
                    img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1)
                else:
                    img = self.cached_image_embeddings[img_name]

                # --- control image load ---
                if self.cached_control_image_embeddings is None:
                    control_img = Image.open(img_path).convert("RGB")
                    control_img = image_resize(control_img, self.img_size)
                    control_img = torch.from_numpy((np.array(control_img) / 127.5) - 1).permute(2, 0, 1)
                else:
                    control_img = self.cached_control_image_embeddings[img_name]

                # --- text embedding ---
                if self.cached_text_embeddings is None:
                    txt_path = os.path.join(os.path.dirname(img_path), txt)
                    if not os.path.exists(txt_path):
                        raise FileNotFoundError(f"Caption file not found: {txt_path}")
                    prompt = open(txt_path, encoding="utf-8").read()
                    if throw_one(self.caption_dropout_rate):
                        return img, " ", control_img
                    return img, prompt, control_img
                else:
                    if throw_one(self.caption_dropout_rate):
                        key = txt + "empty_embedding"
                    else:
                        key = txt
                    if key not in self.cached_text_embeddings:
                        raise KeyError(f"Missing embedding key: {key}")
                    emb = self.cached_text_embeddings[key]
                    return img, emb["prompt_embeds"], emb["prompt_embeds_mask"], control_img

            except Exception as e:
                print(f"Error loading sample (try again): {e}")
                continue
        raise RuntimeError("Too many dataset loading errors")

        

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
