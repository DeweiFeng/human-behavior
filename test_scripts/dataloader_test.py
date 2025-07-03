from dataloaders.daic_loader import DAICWOZDataset
from dataloaders.meld_loader import MELDDataset
from dataloaders.mosei_loader import MOSEIDataset
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional


def test_meld_dataloader(data_dir, split):
    dataset = MELDDataset(data_dir=data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        videos = batch["video"]
        labels = batch["label"]
        filenames = batch["filename"]

        print(type(videos))
        # do something here


def test_mosei_dataloader(data_dir, split):
    def mosei_collate_fn(batch):
        """
        Custom collate function for MOSEI dataset to handle mixed data types. Dataset contains
        tensors and non-tensors(id).
        """
        # Separate tensors and non-tensors
        tensors = {}
        non_tensors = {}

        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                tensors[key] = torch.stack([item[key] for item in batch])
            else:
                non_tensors[key] = [item[key] for item in batch]

        result = {**tensors, **non_tensors}
        return result

    def create_mosei_dataloader(
        data_dir: str,
        split: str = "train",
        modalities: List[str] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:

        dataset = MOSEIDataset(data_dir, split, modalities)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=mosei_collate_fn,
            **kwargs,
        )

    # Check available splits and modalities
    print("Available splits:", MOSEIDataset.get_available_splits(data_dir))
    print("Available modalities:", MOSEIDataset.get_available_modalities(data_dir))

    dataset = MOSEIDataset(
        data_dir, split=split, modalities=["vision", "audio", "text"]
    )
    dataloader = create_mosei_dataloader(data_dir, split=split, batch_size=4)

    # Get sample
    sample = dataset[0]
    print("\nSample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape} ({value.dtype})")
        else:
            print(f"{key}: {value}")

    # Test batch
    for batch in dataloader:
        print("\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {len(value)} items (list)")
        break


if __name__ == "__main__":
    # test_meld_dataloader('/orcd/pool/003/dewei/dataset/meld/MELD.Raw', "test")
    test_mosei_dataloader("datasets/MOSEI", "train")
