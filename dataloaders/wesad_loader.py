import os
import pickle
import numpy as np
import torch
from dataset.template import BaseMultimodalDataset, MultimodalSample
from typing import Dict, List, Tuple, Any, Optional
import glob

class WESADDataset(BaseMultimodalDataset):
    
    LABEL_MAP = {
        0: 'baseline',
        1: 'amusement', 
        2: 'meditation',
        3: 'stress',
        4: 'disgust',
        6: 'sadness',
        7: 'fear'
    }
    
    def __init__(self, data_dir: str, subjects: List[str] = None, modalities: List[str] = None, window_size: int = 1000):
        super().__init__(modality_keys=["physio"])

        self.data_dir = data_dir
        self.subjects = subjects or self._get_available_subjects()
        self.modalities = modalities or ['ecg', 'eda', 'emg', 'resp', 'temp', 'bvp', 'acc_chest', 'acc_wrist']
        self.window_size = window_size

        # Load and process data
        self.samples = self._load_data()
        
        print(f"WESAD dataset loaded:")
        print(f"  Subjects: {len(self.subjects)} ({', '.join(self.subjects)})")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Modalities: {self.modalities}")
        print(f"  Window size: {window_size}")

    def _get_available_subjects(self) -> List[str]:
        """Get all available subject directories."""
        subject_dirs = glob.glob(os.path.join(self.data_dir, "S*"))
        subjects = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]
        return sorted(subjects)
    
    def _load_data(self) -> List[MultimodalSample]:
        """Load data from all subjects and create samples."""
        all_samples = []
        
        for subject in self.subjects:
            subject_samples = self._load_subject_data(subject)
            all_samples.extend(subject_samples)
        
        return all_samples
    
    def _load_subject_data(self, subject: str) -> List[MultimodalSample]:
        """Load data for a single subject and create samples."""
        pickle_path = os.path.join(self.data_dir, subject, f"{subject}.pkl")
        
        if not os.path.exists(pickle_path):
            print(f"Warning: No data found for subject {subject}")
            return []

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        chest_signals = data['signal']['chest']
        wrist_signals = data['signal']['wrist']
        labels = data['label']

        samples = []
        num_windows = len(labels) // self.window_size
        
        for i in range(num_windows):
            start_idx = i * self.window_size
            end_idx = start_idx + self.window_size

            window_labels = labels[start_idx:end_idx]
            majority_label = np.bincount(window_labels).argmax()
            
            sample = MultimodalSample(
                id=f"{subject}_window_{i}",
                physio=None, 
                label=majority_label,
                metadata={
                    'subject': subject,
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'wesad_signals': {
                        'ecg': chest_signals['ECG'][start_idx:end_idx] if 'ecg' in self.modalities else None,
                        'eda': chest_signals['EDA'][start_idx:end_idx] if 'eda' in self.modalities else None,
                        'emg': chest_signals['EMG'][start_idx:end_idx] if 'emg' in self.modalities else None,
                        'resp': chest_signals['Resp'][start_idx:end_idx] if 'resp' in self.modalities else None,
                        'temp': chest_signals['Temp'][start_idx:end_idx] if 'temp' in self.modalities else None,
                        'bvp': wrist_signals['BVP'][start_idx:end_idx] if 'bvp' in self.modalities else None,
                        'acc_chest': chest_signals['ACC'][start_idx:end_idx] if 'acc_chest' in self.modalities else None,
                        'acc_wrist': wrist_signals['ACC'][start_idx:end_idx] if 'acc_wrist' in self.modalities else None,
                    }
                }
            )
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx].to_dict()
    
    @classmethod
    def get_label_name(cls, label: int) -> str:
        """Get the name of a label."""
        return cls.LABEL_MAP.get(label, 'unknown')

    def get_label_distribution(self) -> Dict[str, int]:
        """Get the distribution of labels in the dataset."""
        label_counts = {}
        for sample in self.samples:
            label_name = self.get_label_name(sample.label)
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        return label_counts
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """Get the distribution of subjects in the dataset."""
        subject_counts = {}
        for sample in self.samples:
            subject = sample.metadata['subject']
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        return subject_counts


def wesad_collate_fn(batch):
    """
    Custom collate function for WESAD dataset to handle None values and mixed data types.
    """
    # Separate tensors and non-tensors
    tensors = {}
    non_tensors = {}
    
    for key in batch[0].keys():
        if key == 'metadata':
            # Handle metadata specially
            non_tensors[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], (torch.Tensor, np.ndarray)) or batch[0][key] is not None:
            # Try to stack tensors/arrays, fall back to list if mixed types
            try:
                tensors[key] = torch.stack([torch.tensor(item[key]) if isinstance(item[key], np.ndarray) else item[key] for item in batch])
            except:
                non_tensors[key] = [item[key] for item in batch]
        else:
            non_tensors[key] = [item[key] for item in batch]
    
    # Combine tensors and non-tensors
    result = {**tensors, **non_tensors}
    return result


def create_wesad_dataloader(data_dir: str, subjects: List[str] = None, 
                           modalities: List[str] = None, window_size: int = 1000,
                           batch_size: int = 32, shuffle: bool = True, 
                           num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the WESAD dataset.
    
    Args:
        data_dir: Directory containing WESAD data
        subjects: List of subject IDs to include
        modalities: List of modalities to load
        window_size: Size of time windows
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
    
    Returns:
        PyTorch DataLoader for WESAD dataset
    """
    dataset = WESADDataset(data_dir, subjects, modalities, window_size)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=wesad_collate_fn
    )


if __name__ == "__main__":
    # Example usage
    data_dir = "datasets/WESAD"
    dataloader = create_wesad_dataloader(
        data_dir=data_dir,
        subjects=['S2', 'S3'],  # Use only 2 subjects for testing
        modalities=['ecg', 'eda', 'acc_chest'],
        window_size=1000,
        batch_size=4
    )
    
    print(f"Created dataloader with {len(dataloader.dataset)} samples")
    
    # Test loading a batch
    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Sample ID: {batch['id'][0]}")
        print(f"Label: {batch['label'][0]} ({WESADDataset.get_label_name(batch['label'][0].item())})")
        print(f"ECG shape: {batch['metadata']['wesad_signals']['ecg'].shape}")
        break