# from dataloaders.daic_loader import DAICWOZDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.meld import MELDDataset
from dataloaders.mosei_loader import create_mosei_dataloader
from dataloaders.wesad_loader import create_wesad_dataloader
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional

def test_meld_dataloader(data_dir, split):
    dataset = MELDDataset(data_dir=data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        videos = batch['video']
        labels = batch['label']
        filenames = batch['filename']

        print(type(videos))
        # do something here

def test_wesad_dataloader(data_dir, subjects, modalities, window_size):
    dataloader = create_wesad_dataloader(
        data_dir="datasets/WESAD",
        subjects=['S2', 'S3', 'S4'],
        modalities=['ecg', 'eda', 'acc_chest'],
        window_size=1000,
        batch_size=32
    )

    for batch in dataloader:
        ecg_data = batch['metadata'][0]['wesad_signals']['ecg']
        labels = batch['label']


if __name__ == "__main__":
    # test_meld_dataloader('/orcd/pool/003/dewei/dataset/meld/MELD.Raw', "test")
    test_wesad_dataloader("datasets/WESAD", ['S2'], ['ecg', 'eda'], 1000)