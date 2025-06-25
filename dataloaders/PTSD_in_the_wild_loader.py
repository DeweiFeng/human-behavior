import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample
from moviepy.editor import VideoFileClip

class PTSDITWDataset(BaseMultimodalDataset):
    # Note: the researcher also sent a folder of 3-fold splits that isn't integrated

    TYPE_MAP = {
        'ptsd': 0,
        'no ptsd': 1
    }

    def __init__(self, data_dir: str, split: str):
        super().__init__(modality_keys=["video"])

        self.samples = []
        split_path = os.path.join(data_dir, split)
        for label_name in os.listdir(split_path):
            label_path = os.path.join(split_path, label_name)
            if not os.path.isdir(label_path):
                continue
            label = 0 if label_name == "PTSD" else 1
            for subdir, _, files in os.walk(label_path):
                cause = os.path.basename(os.path.dirname(subdir))
                for file in files:
                    if file.endswith('.mp4'):
                        video_path = os.path.join(subdir, file)
                        base = os.path.splitext(file)[0]
                        utt_id = re.sub(r'[^a-zA-Z0-9]', '_', base).lower()
                        clip = VideoFileClip(file)
                        duration = clip.duration
                        sample = MultimodalSample(
                            id = utt_id, 
                            video=video_path,
                            label=label,
                            metadata={
                                "ptsd_cause": cause, 
                                "video_length": duration
                            }
                        )
                        self.samples.append(sample)


    def __len__(self):
        return len(self.samples)

    # idx refers to the index in video_paths that the video corresponds to
    def __getitem__(self, idx):
        sample = self.samples[idx]

        if isinstance(sample.video, str) and os.path.exists(sample.video):
            with open(sample.video, "rb") as f:
                sample.video = f.read()
        
        return sample.to_dict()


    @classmethod
    def get_diagnosis(cls, label: int) -> str:
        for disorder, idx in cls.TYPE_MAP.items():
            if idx == label:
                return disorder
        return "unknown"
