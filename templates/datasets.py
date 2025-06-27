from typing import Optional, Union, Dict, Any
import numpy as np
from torch.utils.data import Dataset


class MultimodalSample:
    def __init__(
        self,
        id: str,
        text: Optional[str] = None,
        audio: Optional[Union[np.ndarray, str]] = None,
        video: Optional[Union[bytes, str]] = None,
        face: Optional[np.ndarray] = None,
        gesture: Optional[np.ndarray] = None,
        physio: Optional[np.ndarray] = None,
        eeg: Optional[np.ndarray] = None,
        task: Any = None,
        label: Optional[Union[int, float, str, Dict]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.id = id
        self.text = text
        self.audio = audio
        self.video = video
        self.face = face
        self.gesture = gesture
        self.physio = physio
        self.eeg = eeg
        self.label = label
        self.metadata = metadata or {}

    def to_dict(self):
        return self.__dict__


class BaseMultimodalDataset(Dataset):
    def __init__(
        self,
        modality_keys=["text", "audio", "video", "eeg", "physio", "face", "gesture"],
    ):
        self.samples = []
        self.modality_keys = modality_keys

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].to_dict()
