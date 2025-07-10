import os
import pickle
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
from dataset.template import BaseMultimodalDataset, MultimodalSample
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader


class MOSEIDataset(BaseMultimodalDataset):
    """
    MOSEI (Multimodal Opinion Sentiment and Emotion Intensity) Dataset Loader.
    
    The MOSEI dataset contains multimodal data with:
    - Vision: facial features (50 frames, 35 features per frame)
    - Audio: acoustic features (50 frames, 74 features per frame) 
    - Text: linguistic features (50 frames, 300 features per frame)
    - Labels: sentiment/emotion labels
    """
    
    def __init__(self, data_dir: str, split: str = "train", modalities: List[str] = None):
        """
        Initialize MOSEI dataset.
        
        Args:
            data_dir: Directory containing MOSEI data files
            split: Dataset split ('train', 'valid', 'test')
            modalities: List of modalities to load ('vision', 'audio', 'text')
        """
        super().__init__(modality_keys=["vision", "audio", "text"])

        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities or ["vision", "audio", "text"]
        
        # Load the sentiment data
        senti_data_path = os.path.join(data_dir, "mosei_senti_data.pkl")
        if not os.path.exists(senti_data_path):
            raise FileNotFoundError(f"MOSEI sentiment data not found at {senti_data_path}")
        
        with open(senti_data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Check if split exists
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in MOSEI data. Available splits: {list(self.data.keys())}")
        
        # Get split data
        self.split_data = self.data[split]
        self.num_samples = len(self.split_data['id'])
        
        # Create MultimodalSample objects
        self.samples = self._create_sample()
        
        print(f"MOSEI {split} dataset loaded:")
        print(f"  Number of samples: {self.num_samples}")
        print(f"  Modalities: {self.modalities}")
        for modality in self.modalities:
            if modality in self.split_data:
                shape = self.split_data[modality].shape
                print(f"  {modality.capitalize()}: {shape}")
        print(f"  Labels: {self.split_data['labels'].shape}")
    
    def _create_sample(self):
        samples = []
        for i in range(self.num_samples):
            sample = MultimodalSample(
                id=str(self.split_data['id'][i]),
                audio=self.split_data['audio'][i] if 'audio' in self.modalities else None,
                vision=self.split_data['vision'][i] if 'vision' in self.modalities else None,
                text=self.split_data['text'][i] if 'text' in self.modalities else None,
                label=self.split_data['labels'][i],
                metadata={
                    'split': self.split,
                    'modalities': self.modalities,
                }
            )
            samples.append(sample)
        return samples

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx].to_dict()
    
    def get_modality_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get the shapes of each modality."""
        shapes = {}
        for modality in self.modalities:
            if modality in self.split_data:
                shapes[modality] = self.split_data[modality].shape[1:]  # Remove batch dimension
        return shapes
    
    def get_label_info(self) -> Dict[str, Any]:
        """Get information about the labels."""
        if 'labels' not in self.split_data:
            return {}
        
        labels = self.split_data['labels']
        return {
            'shape': labels.shape,
            'min': float(np.min(labels)),
            'max': float(np.max(labels)),
            'mean': float(np.mean(labels)),
            'std': float(np.std(labels))
        }
    
    @classmethod
    def get_available_splits(cls, data_dir: str) -> List[str]:
        """Get available dataset splits."""
        senti_data_path = os.path.join(data_dir, "mosei_senti_data.pkl")
        if not os.path.exists(senti_data_path):
            return []
        
        with open(senti_data_path, 'rb') as f:
            data = pickle.load(f)
        
        return list(data.keys())
    
    @classmethod
    def get_available_modalities(cls, data_dir: str, split: str = "train") -> List[str]:
        """Get available modalities for a given split."""
        senti_data_path = os.path.join(data_dir, "mosei_senti_data.pkl")
        if not os.path.exists(senti_data_path):
            return []
        
        with open(senti_data_path, 'rb') as f:
            data = pickle.load(f)
        
        if split not in data:
            return []
        
        return list(data[split].keys())


def mosei_collate_fn(batch):
    """
    Custom collate function for MOSEI dataset to handle mixed data types. Dataset contains
    tensors and non-tensors(id).
    """
    # Separate tensors and non-tensors
    tensors = {}
    non_tensors = {}
    
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            tensors[key] = torch.stack([item[key] for item in batch])
        else:
            non_tensors[key] = [item[key] for item in batch]
    
    result = {**tensors, **non_tensors}
    return result


def create_mosei_dataloader(
    data_dir: str,
    split: str = "train",
    modalities: List[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    
    dataset = MOSEIDataset(data_dir, split, modalities)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mosei_collate_fn,
        **kwargs
    )


def test_mosei_dataloader(data_dir, split):
    # Check available splits and modalities
    print("Available splits:", MOSEIDataset.get_available_splits(data_dir))
    print("Available modalities:", MOSEIDataset.get_available_modalities(data_dir))
    
    dataset = MOSEIDataset(data_dir, split=split, modalities=['vision', 'audio', 'text'])
    dataloader = create_mosei_dataloader(data_dir, split=split, batch_size=4)

    # Get sample
    sample = dataset[0]
    print("\nSample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape} ({value.dtype})")
        else:
            print(f"{key}: {value}")
    
    # Test batch
    for batch in dataloader:
        print("\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {len(value)} items (list)")
        break