import os
import csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample
import torch.nn.functional as F

class LMVDDataset(BaseMultimodalDataset):
    TYPE_MAP = {
        'depression': 0, 
        'normal': 1
    }

    # Data doesn't include inherent splits
    def __init__(self, data_dir, modalities: List[str] = None):
        super().__init__(modality_keys=["audio", "face"])
        self.modalities = modalities or ["audio", "face"]

        self.data_dir = os.path.expanduser(data_dir)
        audio_dir = os.path.join(self.data_dir, "Audio_feature")
        face_dir = os.path.join(self.data_dir, "Video_feature")

        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        if not os.path.exists(face_dir):
            raise FileNotFoundError(f"Face directory not found: {face_dir}")

        # Only include files with both audio and video features
        self.samples = []
        for f in os.listdir(audio_dir):
            if f.endswith(".npy"):
                face_path = os.path.join(face_dir, f.replace(".npy", ".csv"))
                audio_path = os.path.join(audio_dir, f)

                if os.path.exists(face_path):
                    utt_id = os.path.splitext(f)[0]

                    # Determine label from filename
                    index = int(utt_id)
                    if 1 <= index <= 601 or 1117 <= index <= 1423:
                        label = 0
                    elif 602 <= index <= 1116 or 1425 <= index <= 1824:
                        label = 1
                    else:
                        continue
                    metadata = {}
                    if "face" in self.modalities:
                        with open(face_path, newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            reader.fieldnames = [col.strip() for col in reader.fieldnames]
                            reader = list(reader)
                            if reader:
                                first_row = reader[0]
                                last_row = reader[-1]
                                metadata["speaker"] = first_row["face_id"]
                                start_time = float(first_row.get("timestamp", 0.0))
                                end_time = float(last_row.get("timestamp", 0.0))
                                metadata["duration"] = max(end_time - start_time, 0.0)

                    sample = MultimodalSample(
                        id=utt_id,
                        label=label,
                        metadata=metadata
                    )
                    if "face" in self.modalities:
                        sample.face = face_path
                    if "audio" in self.modalities:
                        sample.audio = audio_path

                    self.samples.append(sample)


    def __len__(self):
        # Audio directory has the same number of files as video
        return len(self.samples) 

    def __getitem__(self, idx):
        # Load audio (.npy)
        sample = self.samples[idx]
        if "audio" in self.modalities and isinstance(sample.audio, str):
            if os.path.exists(sample.audio):
                with open(sample.audio, "rb") as f:
                    # Convert to audio bytes 
                    # sample.audio = f.read()

                    # Load the NP array directly
                    # sample.audio = np.load(sample.audio)

                    # Convert to tensor
                    sample.audio = torch.tensor(np.load(sample.audio), dtype=torch.float32)

        return sample.to_dict()
    
    @classmethod
    def get_diagnosis(cls, label: int) -> str:
        for disorder, idx in cls.TYPE_MAP.items():
            if idx == label:
                return disorder
        return "unknown"
    
def create_LMVD_dataloader(
    data_dir: str,
    split: str = "train",
    modalities: List[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    dataset = LMVDDataset(data_dir, modalities=modalities)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LMVD_collate_fn,
        **kwargs
    )

def LMVD_collate_fn(batch):
    """
    Custom collate function for LMVD dataset.
    Automatically handles mixed data types (tensors, strings, dicts, bytes).
    """
    
    collated = {}
    for key in batch[0].keys():
        first_val = batch[0][key]

        if isinstance(first_val, torch.Tensor):
            # Pad sequences to the same length
            max_len = max(item[key].shape[0] for item in batch)
            padded = [
                F.pad(item[key], (0, 0, 0, max_len - item[key].shape[0]))
                for item in batch
            ]
            collated[key] = torch.stack(padded)
        else:
            collated[key] = [item.get(key, None) for item in batch]
    return collated

def test_LMVD_dataloader(data_dir):
    print("Initializing dataset...")
    dataset = LMVDDataset(data_dir, modalities=['audio', 'face'])
    dataloader = create_LMVD_dataloader(data_dir, modalities=['audio', 'face'], batch_size=4)

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
    # Requires absolute path
    data_dir = "~/LMVD_Feature"
    test_LMVD_dataloader(data_dir)

    
    
