import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class CremaDDataset(Dataset):
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    EMOTION_MAP = {code: idx for idx, code in enumerate(['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])}
    LEVEL_LABELS = ['low', 'medium', 'high', 'unknown']
    LABEL_MAP = {label: idx for idx, label in enumerate(['LO', 'MD', 'HI', 'XX'])}

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_paths = []

        self.audio_files = []
        self.labels = defaultdict(list)
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_name = os.path.splitext(file)[0]
                    path = os.path.join(root, file)
                    self.audio_paths.append(path)
                    label_names = file_name.split('_')

                    # Assume that the actor (1) and the sentence spoken (2) aren't important
                    _, _, emotion, intensity = label_names
                    label = [-1, -1]
                    if emotion in self.EMOTION_MAP:
                        label[0] = self.EMOTION_MAP[emotion]
                    if intensity in self.LABEL_MAP:
                        label[1] = self.LABEL_MAP[intensity]

                    # self.labels should have values in the form of [emotion, intensity]
                    self.labels[file_name] = label

                    self.audio_files.append((path, file_name, label))
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        path, file_name, label = self.audio_files[idx]

        with open(path, 'rb') as f:
            audio_bytes = f.read() # raw audio bytes. Can adjust this line to load audio instead

        return {
            'audio': audio_bytes,
            'label': label,
            'filename': file_name
        }

    @classmethod
    def get_emotion_and_intensity(cls, label) -> str:
        ans = ["unknown", "unknown"]
        for idx, emotion in enumerate(cls.EMOTION_LABELS):
            if idx == label[0]:
                ans[0] = emotion
        for idx, intensity in enumerate(cls.LEVEL_LABELS):
            if idx == label[1]:
                ans[1] = intensity
        return ans


