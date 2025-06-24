import os
import zipfile
import pandas as pd
import csv
from torch.utils.data import Dataset


class DAICWOZDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split_csv,
        label_type="binary",  # or "score"
        modalities=("transcript", "covarep", "clnf_aus", "clnf_pose"),
        exclude_sessions=(342, 394, 398, 460),
    ):
        self.root_dir = root_dir
        self.split_df = pd.read_csv(split_csv)
        self.split_df.columns = self.split_df.columns.str.lower()

        self.subject_ids = self.split_df["participant_id"].astype(str).tolist()
        self.exclude_sessions = set(str(s) for s in exclude_sessions)
        self.subject_ids = [sid for sid in self.subject_ids if sid not in self.exclude_sessions]
        self.modalities = set(modalities)

        # Detect correct label column
        if label_type == "binary":
            label_col = "phq8_binary" if "phq8_binary" in self.split_df.columns else "phq_binary"
        elif label_type == "score":
            label_col = "phq8_score" if "phq8_score" in self.split_df.columns else "phq_score"
        else:
            raise ValueError("label_type must be 'binary' or 'score'")

        self.label_map = dict(zip(self.split_df["participant_id"].astype(str), self.split_df[label_col]))

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        zip_path = os.path.join(self.root_dir, f"{subject_id}_P.zip")
        prefix = f"{subject_id}_"

        sample = {
            "subject_id": subject_id,
            "label": self.label_map.get(subject_id, -1)
        }

        with zipfile.ZipFile(zip_path, "r") as z:
            # Load transcript
            if "transcript" in self.modalities:
                try:
                    with z.open(f"{prefix}TRANSCRIPT.csv") as f:
                        
                        
                        df = pd.read_csv(f, sep="\t")

                        start_times = df["start_time"].to_list()
                        stop_times = df["stop_time"].to_list()
                        speakers = df["speaker"].to_list()
                        values = df["value"].to_list()

                        sample["transcript"] = {
                            "start_times": start_times,
                            "stop_times": stop_times,
                            "speakers": speakers,
                            "values": values,
                        }
                except KeyError:
                    sample["transcript"] = []

            # Load COVAREP
            if "covarep" in self.modalities and f"{prefix}COVAREP.csv" in z.namelist():
                with z.open(f"{prefix}COVAREP.csv") as f:
                    df = pd.read_csv(f)
                    sample["covarep"] = df.fillna(0).to_numpy()

            # Load CLNF AUs
            if "clnf_aus" in self.modalities and f"{prefix}CLNF_AUs.txt" in z.namelist():
                with z.open(f"{prefix}CLNF_AUs.txt") as f:
                    df = pd.read_csv(f)
                    sample["clnf_aus"] = df.fillna(0).to_numpy()

            # Load CLNF Pose
            if "clnf_pose" in self.modalities and f"{prefix}CLNF_pose.txt" in z.namelist():
                with z.open(f"{prefix}CLNF_pose.txt") as f:
                    df = pd.read_csv(f)
                    sample["clnf_pose"] = df.fillna(0).to_numpy()

        return sample
