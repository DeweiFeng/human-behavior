import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample
import torch.nn.functional as F

class RAVDESSDataset(BaseMultimodalDataset):

    EMOTION_MAP = {
        'neutral': 1,
        'calm': 2, 
        'happy': 3, 
        'sad': 4,
        'angry': 5,
        'fearful': 6,
        'disgust': 7, 
        'surprised': 8
    }

    # RAVDESS doesn't provide inherent splits
    def __init__(self, data_dir, modalities):
        # RAVDESS has an audio_visual component outside of video and audio
        super().__init__(modality_keys=["audio", "audio-visual", "video"])
        self.modalities = modalities or ["audio", "audio-visual", "video"]
        self.data_dir = os.path.expanduser(data_dir)
        self.samples = []

        for actor in os.listdir(self.data_dir):
            actor_path = os.path.join(self.data_dir, actor)
            if not os.path.isdir(actor_path):
                continue

            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    utt_id = os.path.splitext(file)[0]
                    audio_path = os.path.join(actor_path, file)

                    # id[0] labels modality
                    ids = utt_id.split("-")
                    if len(ids) < 7:
                        continue # filename is malformed

                    ids_copy = ids.copy()
                    ids_copy[0] = "02"
                    audio_visual_path = os.path.join(actor_path, '-'.join(ids_copy) + ".mp4")
                    if not os.path.exists(audio_visual_path):
                        continue
                
                    ids_copy[0] = "01"
                    video_path = os.path.join(actor_path, '-'.join(ids_copy) + ".mp4")
                    if not os.path.exists(video_path):
                        continue
                
                    vocal_channel = "speech" if ids[1] == "01" else "song"
                    label = int(ids[2])

                    intensity = None if label == 1 else ("normal" if ids[3] == "01" else "strong")
                    text = "Kids are talking by the door" if ids[4] == "01" else "Dogs are sitting by the door"
                    repetition = "1st repetition" if ids[5] == "01" else "2nd repetition"

                    sample = MultimodalSample(
                        id=utt_id,
                        label=label,
                        text=text,
                        metadata={
                            "actor": actor,
                            "vocal channel": vocal_channel,
                            "intensity": intensity,
                            "repetition": repetition
                        }
                    )
                    if "audio" in self.modalities:
                        sample.audio = audio_path
                    if "video" in self.modalities:
                        sample.video = video_path
                    if "audio-visual" in self.modalities:
                        sample.audio_visual = audio_visual_path
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples.")
        if len(self.samples) == 0:
            print("[ERROR] No samples found. Check if path is correct and files are accessible.")
            return

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        for key in ["audio", "video", "audio-visual"]:
            path = getattr(sample, key, None)
            if isinstance(path, str) and os.path.exists(path):
                try:
                    # Try loading as a NumPy array (assumed .npy)
                    np_array = np.load(path)
                    setattr(sample, key, torch.tensor(np_array, dtype=torch.float32))
                except Exception as e:
                    print(f"[WARNING] Failed to load {key} as .npy tensor: {e}. Falling back to bytes.")
                    # Fallback: read as raw bytes (e.g., .mp4)
                    with open(path, "rb") as f:
                        setattr(sample, key, f.read())

        return sample.to_dict()
    
    @classmethod
    def get_emotion_name(cls, label: int) -> str:
        return next((emo for emo, idx in cls.EMOTION_MAP.items() if idx == label), "unknown")
    
def ravdess_collate_fn(batch):
    """
    Specialized collate function for RAVDESSDataset
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            # Pad with 0s
            max_len = max(item[key].shape[0] for item in batch)
            padded = [
                F.pad(item[key], (0, 0, 0, max_len - item[key].shape[0]))
                for item in batch
            ]
            collated[key] = torch.stack(padded)
        elif isinstance(values[0], dict):
            # Handle metadata specially
            collated[key] = {
                subkey: [v[subkey] for v in values] for subkey in values[0]
            }
        else:
            collated[key] = [item.get(key, None) for item in batch]
    return collated

def create_ravdess_dataloader(
    data_dir: str,
    split: str = "train",
    modalities: List[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    dataset = RAVDESSDataset(data_dir, modalities)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ravdess_collate_fn,
        **kwargs
    )

def test_ravdess_dataloader(data_dir):
    print("Initializing dataset...")
    data_dir = os.path.expanduser(data_dir)
    dataset = RAVDESSDataset(data_dir, modalities=['video'])
    dataloader = create_ravdess_dataloader(data_dir, modalities=['video'], batch_size=4)

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

    data_dir = "~/ravdess"
    test_ravdess_dataloader(data_dir)







                



