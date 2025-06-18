import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any

class LMVDDataset(Dataset):
    TYPE_MAP = {
        'depression': 0, 
        'normal': 1
    }
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "Audio_feature")
        self.video_dir = os.path.join(data_dir, "Video_feature")

        # Only include files with both audio and video features
        self.file_list = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.audio_dir)
            if f.endswith(".npy") and os.path.exists(os.path.join(self.video_dir, f.replace(".npy", ".csv")))
        ])


    def __len__(self):
        # Audio directory has the same number of files as video
        return len(self.file_list) 

    def __getitem__(self, idx):
        # Load audio (.npy)
        base_name = self.file_list[idx]
        audio_path = os.path.join(self.audio_dir, base_name + ".npy")
        audio = np.load(audio_path)

        # Load video (.csv)
        video_path = os.path.join(self.video_dir, base_name + ".csv")
        video_df = pd.read_csv(video_path)
        video = video_df.to_numpy()

        # Determine label from filename
        index = int(base_name)
        if 1 <= index <= 601 or 1117 <= index <= 1423:
            label = 0
        elif 602 <= index <= 1116 or 1425 <= index <= 1824:
            label = 1
        else:
            label = -1

        return {
            "audio": audio,
            "video": video, 
            "label": label,
            "id": base_name,
        }
    
    @classmethod
    def get_diagnosis(cls, label: int) -> str:
        for disorder, idx in cls.TYPE_MAP.items():
            if idx == label:
                return disorder
        return "unknown"
