import os
import csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample

class LMVDDataset(BaseMultimodalDataset):
    TYPE_MAP = {
        'depression': 0, 
        'normal': 1
    }

    # Data doesn't include inherent splits
    def __init__(self, data_dir):
        super().__init__(modality_keys=["audio", "face"])
        audio_dir = os.path.join(data_dir, "Audio_feature")
        face_dir = os.path.join(data_dir, "Video_feature")

        # Only include files with both audio and video features
        self.samples = []
        for f in os.listdir(audio_dir):
            if f.endswith(".npy"):
                face_path = os.path.join(face_dir, f.replace(".npy", ".csv"))
                audio_path = os.path.join(audio_dir, f)
                if os.path.exists(face_path):
                    utt_id = os.path.splitext(f)[0]

                    # Determine label from filename
                    index = int(f)
                    if 1 <= index <= 601 or 1117 <= index <= 1423:
                        label = 0
                    elif 602 <= index <= 1116 or 1425 <= index <= 1824:
                        label = 1
                    else:
                        continue

                    with open(face_path, newline='', encoding='utf-8') as f:
                        reader = list(csv.DictReader(f))
                        if reader:
                            first_row = reader[0]
                            last_row = reader[-1]
                            speaker = first_row["face_id"]
                            start_time = float(first_row.get("timestamp", 0.0))
                            end_time = float(last_row.get("timestamp", 0.0))
                            duration = max(end_time - start_time, 0.0)

                    sample = MultimodalSample(
                        id=utt_id,
                        face=face_path,
                        audio=audio_path,
                        label=label,
                        metadata={
                            "speaker": speaker,
                            "audio_length": duration,
                        }
                    )
                    self.samples.append(sample)


    def __len__(self):
        # Audio directory has the same number of files as video
        return len(self.samples) 

    def __getitem__(self, idx):
        # Load audio (.npy)
        sample = self.samples[idx]
        if isinstance(sample.audio, str) and os.path.exists(sample.audio):
            with open(sample.audio, "rb") as f:
                # Convert to audio bytes 
                sample.audio = f.read()

                # Load the NP array directly
                # sample.audio = np.load(sample.audio)

                # Convert to tensor
                # sample.audio = torch.tensor(np.load(sample.audio), dtype=torch.float32)

        return sample.to_dict()
    
    @classmethod
    def get_diagnosis(cls, label: int) -> str:
        for disorder, idx in cls.TYPE_MAP.items():
            if idx == label:
                return disorder
        return "unknown"
