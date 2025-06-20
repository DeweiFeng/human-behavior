import os
import zipfile
import pandas as pd
from typing import List, Dict, Optional
from torch.utils.data import Dataset


class DAICWOZDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split_csv: str,
        split_type: str,
        load_audio: bool = False,
        load_covarep: bool = False,
    ):
        """
        Args:
            root_dir (str): Folder containing all *_P.zip
            split_csv (str): Path to the split CSV file (e.g. train/dev/test)
            load_audio (bool): If True, loads raw audio bytes
            load_covarep (bool): If True, loads COVAREP features as dataframe
        """
        self.root_dir = root_dir
        self.split_df = pd.read_csv(split_csv)

        # Make all column names lowercase
        self.split_df.columns = self.split_df.columns.str.lower()

        # Optionally make all string values lowercase too
        self.split_df = self.split_df.applymap(
            lambda x: x.lower() if isinstance(x, str) else x
        )

        self.subject_ids = self.split_df['participant_id'].astype(str).tolist()
        self.label_map = dict(zip(
            self.split_df['participant_id'].astype(str),
        ))
        self.load_audio = load_audio
        self.load_covarep = load_covarep

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        zip_filename = f"{subject_id}_P.zip"
        zip_path = os.path.join(self.root_dir, zip_filename)

        sample = {
            "subject_id": subject_id,
            "label": self.label_map.get(subject_id, -1),
            "transcript": [],
        }

        with zipfile.ZipFile(zip_path, "r") as z:
            prefix = f"{subject_id}_"

            # Transcript
            with z.open(f"{prefix}TRANSCRIPT.csv") as f:
                df = pd.read_csv(f)
                sample["transcript"] = df["Value"].tolist()

            # Audio (optional)
            if self.load_audio and f"{prefix}AUDIO.wav" in z.namelist():
                with z.open(f"{prefix}AUDIO.wav") as audio_file:
                    sample["audio"] = audio_file.read()

            # COVAREP (optional)
            if self.load_covarep and f"{prefix}COVAREP.csv" in z.namelist():
                with z.open(f"{prefix}COVAREP.csv") as covarep_file:
                    sample["covarep"] = pd.read_csv(covarep_file)

        return sample
