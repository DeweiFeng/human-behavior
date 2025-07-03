import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample
import json
import subprocess
import torch.nn.functional as F
import torchvision

class PTSDITWDataset(BaseMultimodalDataset):

    TYPE_MAP = {
        'ptsd': 0,
        'no ptsd': 1
    }

    def __init__(self, data_dir: str, split: str, modalities):
        super().__init__(modality_keys=["video"])

        self.modalities = modalities or ["video"]
        self.samples = []
        self.data_dir = os.path.expanduser(data_dir)
        new_dir = os.path.join(self.data_dir, "MyDrive", "PTSD_Project_train_validation_test_split")
        split_path = os.path.join(new_dir, split)

        for label_name in os.listdir(split_path):
            label_path = os.path.join(split_path, label_name)
            if not os.path.isdir(label_path):
                continue
            normalized = label_name.lower().strip()
            if normalized == "ptsd":
                label = 0 
            elif normalized == "no ptsd":
                label = 1
            else:
                print(f"[WARNING] Unknown label folder: '{label_name}' — skipping")
                continue

            for subdir, _, files in os.walk(label_path):
                # Only folders PTSD/ contain PTSD Causes
                if label_name.strip().lower() == "ptsd":
                    cause = os.path.basename(subdir)
                else:
                    cause = "none"
                print(f"Causes of PTSD: {cause}")
                for file in files:
                    if file.endswith('.mp4'):
                        video_path = os.path.join(subdir, file)
                        # if os.path.exists(video_path):
                        #     print(f"Found Video: {video_path}")
                        # else:
                        #     print(f"Skipped: File not found: {video_path}")
            
                        base = os.path.splitext(file)[0]
                        utt_id = re.sub(r'[^a-zA-Z0-9]', '_', base).lower()
                        duration = PTSDITWDataset.get_video_duration(video_path)

                        sample = MultimodalSample(
                            id = utt_id, 
                            video=video_path,
                            label=label,
                            metadata={
                                "ptsd_cause": cause, 
                                "video_length": duration
                            }
                        )
                        self.samples.append(sample)
            
        print(f"Loaded {len(self.samples)} samples.")
        if len(self.samples) == 0:
            print("[ERROR] No samples found. Check if path is correct and files are accessible.")
            return


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if isinstance(sample.video, str) and os.path.exists(sample.video):
            try:
                # Load video frames (T, H, W, C) as uint8
                video_frames, _, _ = torchvision.io.read_video(sample.video, pts_unit='sec')
                video_frames = video_frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

                sample.video = video_frames
            except Exception as e:
                raise RuntimeError(f"Failed to load video tensor from {sample.video}: {e}")
        return sample.to_dict()


    @classmethod
    def get_diagnosis(cls, label: int) -> str:
        for disorder, idx in cls.TYPE_MAP.items():
            if idx == label:
                return disorder
        return "unknown"

    @staticmethod
    def get_video_duration(video_path):
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

def ptsditw_collate_fn(batch):
    """
    Specialized Collate Function for PTSDITWDataset
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            max_len = max(v.shape[0] for v in values)
            padded = []
            for v in values:
                pad_len = max_len - v.shape[0]
                pad_dims = (0,) * (2 * (v.ndim - 1)) + (0, pad_len)
                padded_tensor = F.pad(v, pad_dims)
                padded.append(padded_tensor)
            collated[key] = torch.stack(padded)
        elif isinstance(values[0], dict):
            collated[key] = {
                subkey: [v[subkey] for v in values] for subkey in values[0]
            }
        else:
            collated[key] = values
    return collated

def create_ptsditw_dataloader(
    data_dir: str,
    split: str = "train",
    modalities: List[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader: 
    dataset = PTSDITWDataset(data_dir, split=split, modalities=modalities)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ptsditw_collate_fn,
        **kwargs
    )

def test_ptsditw_dataloader(data_dir):
    print("Initializing dataset...")
    dataset = PTSDITWDataset(data_dir, split="train", modalities=['video'])
    dataloader = create_ptsditw_dataloader(data_dir, split="train", modalities=['video'], batch_size=4)

    print("\nSingle sample test:")
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")

    print("\nBatch test from DataLoader:")
    for batch in dataloader:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")
            else:
                print(f"  {key}: {len(val)} items (type: {type(val[0]) if val else 'unknown'})")
        break

if __name__ == "__main__":

    data_dir = "~/drive"
    test_ptsditw_dataloader(data_dir)

