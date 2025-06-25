import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataset.template import BaseMultimodalDataset, MultimodalSample

class CremaDDataset(BaseMultimodalDataset):
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    EMOTION_MAP = {code: idx for idx, code in enumerate(['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'])}
    LEVEL_LABELS = ['low', 'medium', 'high', 'unknown']
    LABEL_MAP = {label: idx for idx, label in enumerate(['LO', 'MD', 'HI', 'XX'])}

    def __init__(self, data_dir):
        super().__init__(modalaity_keys=["audio"])

        self.root_dir = data_dir
        self.samples = []

        for file in os.listdir(data_dir):
            if file.endswith(".wav"):
                file_name = os.path.splitext(file)[0]
                audio_path = os.path.join(data_dir, file)
                label_names = file_name.split('_')

                # Assume that the actor (1) and the sentence spoken (2) aren't important
                actor, sentence, emotion, intensity = label_names
                if emotion not in self.EMOTION_MAP:
                    continue
                emo_label = self.EMOTION_MAP[emotion]
                if intensity not in self.LABEL_MAP:
                    continue
                int_label = self.LABEL_MAP[intensity]

                sample = MultimodalSample(
                    id=file_name,
                    audio=audio_path,
                    label=emo_label,
                    metadata={
                        "speaker": actor,
                        "sentence": sentence,
                        "intensity": int_label
                    }
                )
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if isinstance(sample.audio, str) and os.path.exists(sample.audio):
            with open(sample.audio, "rb") as f:
                sample.audio = f.read()

        return sample.to_dict()

    @classmethod
    def get_emotion(cls, emo_label) -> str:
        for idx, emotion in enumerate(cls.EMOTION_LABELS):
            if idx == emo_label:
                return emotion
        return "unknown"
    
    @classmethod
    def get_intensity(cls, int_label) -> str:
        for idx, intensity in enumerate(cls.LEVEL_LABELS):
            if idx == int_label:
                return intensity
        return "unknown"


