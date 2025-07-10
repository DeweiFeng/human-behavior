import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
from dataset.template import BaseMultimodalDataset, MultimodalSample
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader


class ESConvDataset(BaseMultimodalDataset):
    def __init__(self, data_dir: str, split: str = "train", modalities: List[str] = None):
        """
        Initialize ESConv dataset.
        
        Args:
            data_dir: Directory containing ESConv data files
            split: Dataset split ('train', 'valid', 'test') - will be used for data splitting
            modalities: List of modalities to load ('text', 'dialog', 'metadata')
        """
        super().__init__(modality_keys=["text", "dialog", "metadata"])

        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities or ["text", "dialog", "metadata"]
        
        # Load the ESConv data
        data_path = os.path.join(data_dir, "ESConv.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"ESConv data not found at {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Split the data based on the split parameter
        self.split_data = self._split_data(self.data, split)
        self.num_samples = len(self.split_data)
        
        # Create MultimodalSample objects
        self.samples = self._create_samples()
        
        print(f"ESConv {split} dataset loaded:")
        print(f"  Number of samples: {self.num_samples}")
        print(f"  Modalities: {self.modalities}")
        print(f"  Total available samples: {len(self.data)}")
    
    def _split_data(self, data: List[Dict], split: str) -> List[Dict]:
        """
        Split the data into train/valid/test sets.
        Using a simple 80/10/10 split based on index.
        """
        total_samples = len(data)
        
        if split == "train":
            return data[:int(0.8 * total_samples)]
        elif split == "valid":
            return data[int(0.8 * total_samples):int(0.9 * total_samples)]
        elif split == "test":
            return data[int(0.9 * total_samples):]
        else:
            raise ValueError(f"Split '{split}' not supported. Use 'train', 'valid', or 'test'")
    
    def _create_samples(self):
        samples = []
        for i, sample_data in enumerate(self.split_data):
            text_content = self._extract_text_content(sample_data)
            
            labels = {
                "emotion_type": sample_data.get("emotion_type"),
                "problem_type": sample_data.get("problem_type"),
                "experience_type": sample_data.get("experience_type"),
                "survey_score": sample_data.get("survey_score", {})
            }
            
            metadata = {
                "split": self.split,
                "modalities": self.modalities,
                "situation": sample_data.get("situation"),
                "seeker_question1": sample_data.get("seeker_question1"),
                "seeker_question2": sample_data.get("seeker_question2"),
                "supporter_question1": sample_data.get("supporter_question1"),
                "supporter_question2": sample_data.get("supporter_question2"),
            }
            
            sample = MultimodalSample(
                id=str(i),
                text=text_content if "text" in self.modalities else None,
                label=labels,
                metadata=metadata
            )
            
            if "dialog" in self.modalities:
                sample.dialog = sample_data.get("dialog", [])
            
            samples.append(sample)
        return samples
    
    def _extract_text_content(self, sample_data: Dict) -> str:
        """
        Extract and concatenate all text content from the dialog.
        """
        dialog = sample_data.get("dialog", [])
        text_parts = []
        
        for turn in dialog:
            content = turn.get("content", "").strip()
            if content:
                speaker = turn.get("speaker", "unknown")
                text_parts.append(f"{speaker}: {content}")
        
        return " ".join(text_parts)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx].to_dict()
    
    def get_emotion_types(self) -> List[str]:
        """Get all unique emotion types in the dataset."""
        emotion_types = set()
        for sample_data in self.split_data:
            emotion_type = sample_data.get("emotion_type")
            if emotion_type:
                emotion_types.add(emotion_type)
        return sorted(list(emotion_types))
    
    def get_problem_types(self) -> List[str]:
        """Get all unique problem types in the dataset."""
        problem_types = set()
        for sample_data in self.split_data:
            problem_type = sample_data.get("problem_type")
            if problem_type:
                problem_types.add(problem_type)
        return sorted(list(problem_types))
    
    def get_dialog_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dialogs."""
        total_turns = 0
        total_seeker_turns = 0
        total_supporter_turns = 0
        dialog_lengths = []
        
        for sample_data in self.split_data:
            dialog = sample_data.get("dialog", [])
            dialog_lengths.append(len(dialog))
            total_turns += len(dialog)
            
            for turn in dialog:
                speaker = turn.get("speaker", "")
                if speaker == "seeker":
                    total_seeker_turns += 1
                elif speaker == "supporter":
                    total_supporter_turns += 1
        
        return {
            "total_dialogs": len(self.split_data),
            "total_turns": total_turns,
            "total_seeker_turns": total_seeker_turns,
            "total_supporter_turns": total_supporter_turns,
            "avg_dialog_length": np.mean(dialog_lengths),
            "min_dialog_length": np.min(dialog_lengths),
            "max_dialog_length": np.max(dialog_lengths),
        }
    
    @classmethod
    def get_available_splits(cls) -> List[str]:
        """Get available dataset splits."""
        return ["train", "valid", "test"]
    
    @classmethod
    def get_available_modalities(cls) -> List[str]:
        """Get available modalities."""
        return ["text", "dialog", "metadata"]


def esconv_collate_fn(batch):
    """
    Custom collate function for ESConv dataset to handle mixed data types.
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


def create_esconv_dataloader(
    data_dir: str,
    split: str = "train",
    modalities: List[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    
    dataset = ESConvDataset(data_dir, split, modalities)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=esconv_collate_fn,
        **kwargs
    )


def test_esconv_dataloader(data_dir, split="train"):
    """Test function for ESConv dataloader."""
    print("Available splits:", ESConvDataset.get_available_splits())
    print("Available modalities:", ESConvDataset.get_available_modalities())
    
    dataset = ESConvDataset(data_dir, split=split, modalities=['text', 'dialog', 'metadata'])
    dataloader = create_esconv_dataloader(data_dir, split=split, batch_size=4)

    sample = dataset[0]
    print("\nSample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape} ({value.dtype})")
        elif isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
    
    print("\nDialog Statistics:")
    stats = dataset.get_dialog_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nEmotion Types:", dataset.get_emotion_types())
    print("Problem Types:", dataset.get_problem_types())
    
    # Test batch
    for batch in dataloader:
        print(f"\nBatch keys: {list(batch.keys())}")
        print(f"Batch size: {len(batch['id'])}")
        break


if __name__ == "__main__":
    data_dir = "datasets/ESConv"
    test_esconv_dataloader(data_dir, "train") 