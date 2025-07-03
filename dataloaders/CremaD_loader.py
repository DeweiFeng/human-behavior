import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataset.template import BaseMultimodalDataset, MultimodalSample
import torch.nn.functional as F

class CremaDDataset(BaseMultimodalDataset):
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    EMOTION_MAP = {code: idx for idx, code in enumerate(['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])}
    LEVEL_LABELS = ['low', 'medium', 'high', 'unknown']
    LABEL_MAP = {label: idx for idx, label in enumerate(['LO', 'MD', 'HI', 'XX'])}

    def __init__(self, data_dir, modalities: List[str] = None):

        super().__init__(modality_keys=["audio"])
        self.modalities = modalities or ["audio"]
        self.samples = []

        self.data_dir = os.path.expanduser(data_dir)

        for file in os.listdir(self.data_dir):
            if file.endswith(".wav"):
                file_name = os.path.splitext(file)[0]
                audio_path = os.path.join(self.data_dir, file)
                label_names = file_name.split('_')

                # Assume that the actor (1) and the sentence spoken (2) aren't important
                actor, sentence, emotion, intensity = label_names
                if emotion not in self.EMOTION_MAP:
                    continue
                emo_label = self.EMOTION_MAP[emotion]
                if intensity not in self.LABEL_MAP:
                    continue
                int_label = self.LABEL_MAP[intensity]

                sample = MultimodalSample(
                    id=file_name,
                    audio=audio_path,
                    label=emo_label,
                    text=sentence,
                    metadata={
                        "speaker": actor,
                        "intensity": int_label
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
        if isinstance(sample.audio, str) and os.path.exists(sample.audio):
            try: 
                np_array = np.load(sample.audio)
                sample.audio = torch.tensor(np_array, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(f"Failed to load audio tensor from {sample.audio}: {e}")

        return sample.to_dict()

    @classmethod
    def get_emotion(cls, emo_label) -> str:
        for idx, emotion in enumerate(cls.EMOTION_LABELS):
            if idx == emo_label:
                return emotion
        return "unknown"
    
    @classmethod
    def get_intensity(cls, int_label) -> str:
        for idx, intensity in enumerate(cls.LEVEL_LABELS):
            if idx == int_label:
                return intensity
        return "unknown"
    
def create_cremad_dataloader(
    data_dir: str,
    split: str = "train",
    modalities: List[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    dataset = CremaDDataset(data_dir, modalities=modalities)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=cremad_collate_fn,
        **kwargs
    )

def cremad_collate_fn(batch):
    """
    Custom collate function for CREMA-D dataset.
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            # Pad sequences to the same length
            max_len = max(v.shape[0] for v in values)
            # Assumes audio tensors are 2D
            padded = [
                F.pad(item[key], (0, 0, 0, max_len - item[key].shape[0]))
                for item in batch
            ]
            collated[key] = torch.stack(padded)
        elif isinstance(values[0], dict):
            collated[key] = {
                subkey: [v[subkey] for v in values] for subkey in values[0]
            }
        else:
            collated[key] = values
    return collated

def test_cremad_dataloader(data_dir):
    print("Initializing dataset...")
    dataset = CremaDDataset(data_dir, modalities=['audio'])
    dataloader = create_cremad_dataloader(data_dir, modalities=['audio'], batch_size=4)

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

    data_dir = "~/AudioWAV"
    test_cremad_dataloader(data_dir)

    
    




