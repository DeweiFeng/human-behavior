import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any

class PTSDITWDataset(Dataset):

    # Note: the researcher also sent a folder of 3-fold splits that isn't integrated
    # with this.

    TYPE_MAP = {
        'PTSD': 0,
        'normal': 1
    }

    def __init__(self, root_dir, transform=None):
        self.video_paths = []
        self.labels = []
        self.file_names = []
        self.transform = transform
        self._load_videos(root_dir)
    
    def _load_videos(self, root_dir):
        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            label = 1 if label_name == "PTSD" else 0
            for subdir, _, files in os.walk(label_path):
                for file in files:
                    if file.endswith('.mp4'):
                        self.video_paths.append(os.path.join(subdir, file))
                        self.labels.append(label)
                        self.file_names.append(file.replace('.mp4', ''))

    def __len__(self):
        return len(self.video_paths)

    # idx refers to the index in video_paths that the video corresponds to
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        filename = self.file_names[idx]

        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        
        return {
            'video': video_bytes, # raw bytes, you may later decode this with torchvision.io or decord
            'label': label, 
            'filename': filename
        }


    @classmethod
    def get_diagnosis(cls, label: int) -> str:
        for disorder, idx in cls.TYPE_MAP.items():
            if idx == label:
                return disorder
        return "unknown"
