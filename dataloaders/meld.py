import os
import csv
from dataset.template import BaseMultimodalDataset, MultimodalSample


class MELDDataset(BaseMultimodalDataset):
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
        super().__init__(modality_keys=["text", "video"])
        self.video_dir = os.path.join(data_dir, f"{split}/{split}_splits")
        self.csv_path = os.path.join(data_dir, f"{split}_sent_emo.csv")

        self.samples = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dia = row["Dialogue_ID"].strip()
                utt = row["Utterance_ID"].strip()
                utt_id = f"dia{dia}_utt{utt}"
                emotion = row["Emotion"].strip().lower()
                label = self.EMOTION_MAP.get(emotion, -1)
                text = row["Utterance"].strip()

                video_path = os.path.join(self.video_dir, f"{utt_id}.mp4")
                if os.path.exists(video_path):
                    sample = MultimodalSample(
                        id=utt_id,
                        text=text,
                        video=video_path,  # video file path (can be lazy-loaded)
                        label=label,
                        metadata={
                            "speaker": row["Speaker"].strip(),
                            "sentiment": row["Sentiment"].strip().lower(),
                            "start_time": row["StartTime"],
                            "end_time": row["EndTime"]
                        }
                    )
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video bytes from path if not already loaded
        if isinstance(sample.video, str) and os.path.exists(sample.video):
            with open(sample.video, "rb") as f:
                sample.video = f.read()

        return sample.to_dict()

    @classmethod
    def get_emotion_name(cls, label: int) -> str:
        return next((emo for emo, idx in cls.EMOTION_MAP.items() if idx == label), "unknown")
