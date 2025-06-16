import os
import csv
import tarfile
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader


class MELDDataset(Dataset):
    EMOTION_MAP = {
        'neutral': 0,
        'surprise': 1,
        'fear': 2,
        'sadness': 3,
        'joy': 4,
        'disgust': 5,
        'anger': 6
    }

    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split

        # Resolve tar.gz and csv paths
        if split == "test":
            self.tar_path = os.path.join(data_dir, "test.tar.gz")
            self.csv_path = os.path.join(data_dir, "test_sent_emo.csv")
        elif split == "dev":
            self.tar_path = os.path.join(data_dir, "dev.tar.gz")
            self.csv_path = os.path.join(data_dir, "dev_sent_emo.csv")
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load label mapping
        self.label_map = {}
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dia = row["Dialogue_ID"].strip()
                utt = row["Utterance_ID"].strip()
                file_key = f"dia{dia}_utt{utt}"
                emotion = row["Emotion"].strip().lower()
                if emotion in self.EMOTION_MAP:
                    self.label_map[file_key] = self.EMOTION_MAP[emotion]
                else:
                    self.label_map[file_key] = -1  # Unknown

        # Index video files inside the tar.gz
        self.video_files = []
        with tarfile.open(self.tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".mp4"):
                    filename = os.path.basename(member.name).replace(".mp4", "")
                    label = self.label_map.get(filename, -1)
                    self.video_files.append((member.name, filename, label))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        tar_member_name, filename, label = self.video_files[idx]

        # Read video bytes on demand
        with tarfile.open(self.tar_path, "r:gz") as tar:
            member = tar.getmember(tar_member_name)
            video_file = tar.extractfile(member)
            video_bytes = video_file.read()

        sample = {
            'video': video_bytes,  # raw bytes, you may later decode this with torchvision.io or decord
            'label': label,
            'filename': filename
        }
        return sample

    @classmethod
    def get_emotion_name(cls, label: int) -> str:
        for emotion, idx in cls.EMOTION_MAP.items():
            if idx == label:
                return emotion
        return "unknown"
