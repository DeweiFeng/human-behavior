import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from dataset.template import BaseMultimodalDataset, MultimodalSample

class ravdess_loader(BaseMultimodalDataset):

    EMOTION_MAP = {
        'neutral': 1,
        'calm': 2, 
        'happy': 3, 
        'sad': 4,
        'angry': 5,
        'fearful': 6,
        'disgust': 7, 
        'surprised': 8
    }

    # RAVDESS doesn't provide inherent splits
    def __init__(self, data_dir):
        super().__init__(modality_keys=["audio", "audio-visual", "video"])
        self.data_dir = data_dir
        self.samples = []

        for actor in os.listdir(data_dir):
            actor_path = os.path.join(data_dir, actor)
            if not os.path.isdir(actor_path):
                continue

            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    utt_id = os.path.splitext(file)[0]
                    audio_path = os.path.join(actor_path, file)

                    # id[0] labels modality
                    ids = utt_id.split("-")
                    if len(ids) < 7:
                        continue # filename is malformed

                    ids_copy = ids.copy()
                    ids_copy[0] = "02"
                    audio_visual_path = os.path.join(actor_path, '-'.join(ids))
                    if not os.path.exists(audio_visual_path):
                        continue
                
                    ids_copy[0] = "01"
                    video_path = os.path.join(actor_path, '-'.join(ids))
                    if not os.path.exists(video_path):
                        continue
                
                    vocal_channel = "speech" if id[1] == "01" else "song"
                    label = int(id[2])

                    intensity = None if label == 1 else ("normal" if ids[3] == "01" else "strong")
                    sentence = "Kids are talking by the door" if ids[4] == "01" else "Dogs are sitting by the door"
                    repetition = "1st repetition" if ids[5] == "01" else "2nd repetition"

                    sample = MultimodalSample(
                        id=utt_id,
                        audio=audio_path,
                        video=video_path,
                        audio_visual=audio_visual_path,
                        label=label,
                        metadata={
                            "actor": actor,
                            "vocal channel": vocal_channel,
                            "intensity": intensity,
                            "sentence": sentence,
                            "repetition": repetition
                        }
                    )
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)
    
    def __get__(self, idx):
        sample = self.samples[idx]

        for key in ["audio", "video", "audio_visual"]:
            path = getattr(sample, key)
            if isinstance(path, str) and os.path.exists(path):
                with open(path, "rb") as f:
                    setattr(sample, key, f.read())

        return sample.to_dict()
    
    @classmethod
    def get_emotion_name(cls, label: int) -> str:
        return next((emo for emo, idx in cls.EMOTION_MAP.items() if idx == label), "unknown")





                



