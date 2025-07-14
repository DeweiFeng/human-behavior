import os
import pickle
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from dataset.template import BaseMultimodalDataset, MultimodalSample
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime

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


def mosei_training(
    data_dir: str,
    train_split: str = "train",
    val_split: str = "valid",
    modalities: List[str] = None,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = None,
    num_workers: int = 4,
    save_path: str = "checkpoints/mosei_model.pt",
    verbose: bool = True
):
    """
    MOSEI training function that can be integrated with train_script.py.
    
    This function sets up MOSEI-specific data loaders and returns them
    along with any MOSEI-specific configurations needed for training.
    
    Args:
        data_dir: Directory containing MOSEI data files
        train_split: Training split name
        val_split: Validation split name
        modalities: List of modalities to use ('vision', 'audio', 'text')
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to train on ('cuda', 'cpu', or None for auto)
        num_workers: Number of data loader workers
        save_path: Path to save the trained model
        verbose: Whether to print training progress
    
    Returns:
        Dictionary containing:
        - train_loader: Training data loader
        - val_loader: Validation data loader
        - dataset_info: Information about the dataset
        - training_config: Configuration for training
    """
    
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Set default modalities if not provided
    if modalities is None:
        modalities = ["vision", "audio", "text"]
    
    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if verbose:
        print(f"MOSEI Training Setup")
        print(f"Device: {device}")
        print(f"Modalities: {modalities}")
        print(f"Data directory: {data_dir}")
    
    # Load datasets
    if verbose:
        print("Loading MOSEI datasets...")
    
    train_dataset = MOSEIDataset(data_dir, train_split, modalities)
    val_dataset = MOSEIDataset(data_dir, val_split, modalities)
    
    # Create dataloaders
    train_loader = create_mosei_dataloader(
        data_dir, train_split, modalities, batch_size, 
        shuffle=True, num_workers=num_workers
    )
    val_loader = create_mosei_dataloader(
        data_dir, val_split, modalities, batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    if verbose:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # Get dataset information
    modality_shapes = train_dataset.get_modality_shapes()
    label_info = train_dataset.get_label_info()
    
    if verbose:
        print("Modality shapes:", modality_shapes)
        print("Label info:", label_info)
    
    # Prepare dataset info for the main training script
    dataset_info = {
        'name': 'MOSEI',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'modalities': modalities,
        'modality_shapes': modality_shapes,
        'label_info': label_info,
        'task_type': 'regression',  # MOSEI is regression for sentiment scores
        'num_classes': 1  # Single output for regression
    }
    
    # Prepare training configuration
    training_config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'device': device,
        'save_path': save_path,
        'criterion': nn.MSELoss(),  # MSE loss for regression
        'optimizer_class': optim.AdamW,
        'scheduler_class': optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'mode': 'min', 'factor': 0.5, 'patience': 5},
        'gradient_clip_val': 1.0,
        'early_stopping_patience': 10
    }
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'dataset_info': dataset_info,
        'training_config': training_config
    }


def mosei_process_batch(batch, device):
    """
    Process a MOSEI batch for training.
    
    This function converts MOSEI batch data into the format expected
    by the MultiTaskMentalHealthModel or other models.
    
    Args:
        batch: Batch from MOSEI dataloader
        device: Device to move tensors to
    
    Returns:
        Dictionary with processed inputs and labels
    """
    import torch
    import numpy as np
    
    # Move batch to device
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value
    
    # Get labels
    labels = batch_device['label'].float()
    
    # Process modalities for the model
    processed_inputs = {
        'vision': None,
        'audio': None,
        'text': None,
        'labels': labels
    }
    
    # Process vision (facial features)
    if 'vision' in batch_device and batch_device['vision'] is not None:
        vision_data = batch_device['vision']
        # Average over time dimension if present
        if len(vision_data.shape) == 3:  # [batch, time, features]
            vision_features = vision_data.mean(dim=1)  # [batch, features]
        else:
            vision_features = vision_data
        processed_inputs['vision'] = vision_features
    
    # Process audio (acoustic features)
    if 'audio' in batch_device and batch_device['audio'] is not None:
        audio_data = batch_device['audio']
        # Average over time dimension if present
        if len(audio_data.shape) == 3:  # [batch, time, features]
            audio_features = audio_data.mean(dim=1)  # [batch, features]
        else:
            audio_features = audio_data
        processed_inputs['audio'] = audio_features
    
    # Process text (linguistic features)
    if 'text' in batch_device and batch_device['text'] is not None:
        text_data = batch_device['text']
        # Average over time dimension if present
        if len(text_data.shape) == 3:  # [batch, time, features]
            text_features = text_data.mean(dim=1)  # [batch, features]
        else:
            text_features = text_data
        processed_inputs['text'] = text_features
    
    return processed_inputs


def mosei_evaluate(model, val_loader, device, criterion):
    """
    Evaluate MOSEI model performance.
    
    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        device: Device to run evaluation on
        criterion: Loss function
    
    Returns:
        Dictionary with evaluation metrics
    """
    import torch
    
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Process batch
            processed_inputs = mosei_process_batch(batch, device)
            labels = processed_inputs['labels']
            
            # Forward pass
            outputs = model(processed_inputs)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            mae = torch.mean(torch.abs(outputs - labels))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return {
        'val_loss': avg_loss,
        'val_mae': avg_mae
    }


