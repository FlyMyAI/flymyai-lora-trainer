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
    def __init__(self, img_dir, img_size=512, img_repeats=10, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1, cached_text_embeddings=None,
                 cached_image_embeddings=None):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_repeats = img_repeats
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        print('cached_text_embeddings', type(cached_text_embeddings))

    def __len__(self):
        """Return the actual number of images in the dataset."""
        return len(self.images) * self.img_repeats

    def __getitem__(self, idx):
        """Get item by index, with looping."""
        idx = idx % len(self.images) # does not include repeats because of the modulo operation

        try:
            # Use the actual index instead of random selection
            img_path = self.images[idx]
            if self.cached_image_embeddings is None:
                img = Image.open(img_path).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        img = crop_to_aspect_ratio(img, ratio)
                img = image_resize(img, self.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1)
            else:
                img = self.cached_image_embeddings[img_path.split('/')[-1]]
            txt_path = img_path.split('.')[0] + '.' + self.caption_type
            if self.cached_text_embeddings is None:
                prompt = open(txt_path).read()
                if throw_one(self.caption_dropout_rate):
                    return img, " "
                else:
                    return img, prompt
            else:
                txt = txt_path.split('/')[-1]
                if throw_one(self.caption_dropout_rate):
                    return img, self.cached_text_embeddings['empty_embedding']['prompt_embeds'], self.cached_text_embeddings['empty_embedding']['prompt_embeds_mask']
                else:
                    return img, self.cached_text_embeddings[txt]['prompt_embeds'], self.cached_text_embeddings[txt]['prompt_embeds_mask']
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a fallback sample instead of infinite recursion
            if idx > 0:
                return self.__getitem__(idx - 1)
            else:
                raise

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
