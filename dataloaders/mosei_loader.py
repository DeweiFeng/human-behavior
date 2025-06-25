import os
import pickle
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional


class MOSEIDataset(Dataset):
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
        self.data_dir = data_dir
        self.split = split
        
        if modalities is None:
            modalities = ['vision', 'audio', 'text']
        self.modalities = modalities
        
        # Load the sentiment data
        senti_data_path = os.path.join(data_dir, "mosei_senti_data.pkl")
        if not os.path.exists(senti_data_path):
            raise FileNotFoundError(f"MOSEI sentiment data not found at {senti_data_path}")
        
        with open(senti_data_path, 'rb') as f:
            data = pickle.load(f)
        
        if split not in data:
            raise ValueError(f"Split '{split}' not found in MOSEI data. Available splits: {list(data.keys())}")
        
        self.data = data[split]
        self.num_samples = len(self.data['id'])
        
        print(f"MOSEI {split} dataset loaded:")
        print(f"  Number of samples: {self.num_samples}")
        print(f"  Modalities: {self.modalities}")
        for modality in self.modalities:
            if modality in self.data:
                shape = self.data[modality].shape
                print(f"  {modality.capitalize()}: {shape}")
        print(f"  Labels: {self.data['labels'].shape}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        sample = {}
        
        # Add modalities
        for modality in self.modalities:
            if modality in self.data:
                data = self.data[modality][idx]
                # Convert to torch tensor
                sample[modality] = torch.tensor(data, dtype=torch.float32)
        
        # Add labels
        if 'labels' in self.data:
            labels = self.data['labels'][idx]
            sample['labels'] = torch.tensor(labels, dtype=torch.float32)
        
        # Add ID information
        if 'id' in self.data:
            sample['id'] = self.data['id'][idx]
        
        return sample
    
    def get_modality_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get the shapes of each modality."""
        shapes = {}
        for modality in self.modalities:
            if modality in self.data:
                shapes[modality] = self.data[modality].shape[1:]  # Remove batch dimension
        return shapes
    
    def get_label_info(self) -> Dict[str, Any]:
        """Get information about the labels."""
        if 'labels' not in self.data:
            return {}
        
        labels = self.data['labels']
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




if __name__ == "__main__":
    # Example usage
    pass