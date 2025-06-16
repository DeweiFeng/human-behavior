import os
import csv
import tarfile
from typing import Dict, List, Tuple, Any


class VideoDatasetLoader:
    # Emotion label mapping
    EMOTION_MAP = {
        'neutral': 0,
        'surprise': 1,
        'fear': 2,
        'sadness': 3,
        'joy': 4,
        'disgust': 5,
        'anger': 6
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.test_tar_path = os.path.join(data_dir, "test.tar.gz")
        self.dev_tar_path = os.path.join(data_dir, "dev.tar.gz")
        self.test_csv_path = os.path.join(data_dir, "test_sent_emo.csv")
        self.dev_csv_path = os.path.join(data_dir, "dev_sent_emo.csv")

    def load_split(self, split: str) -> List[Dict[str, Any]]:
        """Load a specific split of the dataset (test or dev)."""
        if split == "test":
            tar_path = self.test_tar_path
            csv_path = self.test_csv_path
        elif split == "dev":
            tar_path = self.dev_tar_path
            csv_path = self.dev_csv_path
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'test' or 'dev'.")

        # Load labels from CSV
        label_map = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dia = row["Dialogue_ID"].strip()
                utt = row["Utterance_ID"].strip()
                file_key = f"dia{dia}_utt{utt}"
                # Convert emotion text to numeric label
                emotion = row["Emotion"].strip().lower()
                if emotion in self.EMOTION_MAP:
                    label_map[file_key] = self.EMOTION_MAP[emotion]
                else:
                    print(f"Warning: Unknown emotion '{emotion}' for file {file_key}")
                    label_map[file_key] = -1  # Unknown emotion

        # Load videos from tar.gz
        examples = []
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".mp4"):
                    continue

                filename = os.path.basename(member.name).replace(".mp4", "")
                label = label_map.get(filename, -1)

                video_file = tar.extractfile(member)
                video_bytes = video_file.read()

                examples.append({
                    "video": video_bytes,
                    "label": label,
                    "filename": filename
                })

        return examples

    def get_test_split(self) -> List[Dict[str, Any]]:
        """Get the test split."""
        return self.load_split("test")

    def get_dev_split(self) -> List[Dict[str, Any]]:
        """Get the dev split."""
        return self.load_split("dev")

    @classmethod
    def get_emotion_name(cls, label: int) -> str:
        """Convert numeric label back to emotion name."""
        for emotion, idx in cls.EMOTION_MAP.items():
            if idx == label:
                return emotion
        return "unknown"