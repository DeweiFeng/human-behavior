import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample
import pickle
import torch.nn.functional as F

class CH_SIMSv2Dataset(BaseMultimodalDataset):

    SENTIMENT_MAP = {
        'positive': 1,
        'neutral': 0, 
        'negative': -1
    }
    
    def __init__(self, data_dir: str, split: str):
        super().__init__(modality_keys=["text", "audio", "face"])
        self.data_dir = data_dir
        self.split = split

        pickle_path = os.path.join(data_dir, "CH-SIMS v2(s)", "Processed", "unaligned.pkl")

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        split_data = data[split]
        self.data = split_data

        self.samples = []
        for idx in range(len(split_data['id'])):
            id = split_data['id'][idx]
            text = torch.tensor(split_data['raw_text'][idx], dtype=torch.float32)
            audio = torch.tensor(split_data['audio'][idx], dtype=torch.float32)
            vision =  torch.tensor(split_data['vision'][idx], dtype=torch.float32)
            annotation = split_data['annotations'][idx].strip().lower()
            label = self.SENTIMENT_MAP.get(annotation, 0)
            reg_label = torch.tensor(split_data['regression_labels'][idx])
            audio_len =  int(split_data['audio_lengths'][idx])
            vision_len = int(split_data['vision_lengths'][idx])

            sample = MultimodalSample(
                id=id,
                text=text,
                audio=audio,
                face=vision,
                label=label,
                metadata={
                    'audio_length': audio_len,
                    'vision_length': vision_len,
                    'regression_label': reg_label
                }
            )
            self.samples.append(sample)
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        return sample.to_dict()

        
    @classmethod
    def get_sentiment_name(cls, label: float) -> str:
        return next((sen for sen, idx in cls.SENTIMENT_MAP.items() if idx == label), "unknown")
    
def ch_simsv2_collate_fn(batch):
    """
    Custom collate function for CH_SIMSv2 dataset to handle None values and mixed data types.
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            max_len = max(v.shape[0] for v in values)
            padded = []
            for v in values:
                pad_len = max_len - v.shape[0]
                if v.ndim == 1:
                    padded_tensor = F.pad(v, (0, pad_len))
                elif v.ndim == 2:
                    padded_tensor = F.pad(v, (0, 0, 0, pad_len))
                else:
                    raise ValueError(f"Unexpected tensor shape for key '{key}': {v.shape}")
                padded.append(padded_tensor)
            collated[key] = torch.stack(padded)
        # Handle metadata separately
        elif isinstance(values[0], dict):
            collated[key] = {
                subkey: [v[subkey] for v in values] for subkey in values[0]
            }
        else:
            collated[key] = values
    return collated

def create_ch_simsv2_dataloader(
    data_dir: str,
    split: str = "train",
    modalities: List[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    dataset = CH_SIMSv2Dataset(data_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ch_simsv2_collate_fn,
        **kwargs
    )

def test_ch_simsv2_dataloader(data_dir):
    print("Initializing dataset...")
    dataset = CH_SIMSv2Dataset(data_dir, modalities=["text", "audio", "face"])
    dataloader = create_ch_simsv2_dataloader(data_dir, modalities=["text", "audio", "face"], batch_size=4)

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
    data_dir = "~/unaligned.pkl"
    test_ch_simsv2_dataloader(data_dir)

    
    










    