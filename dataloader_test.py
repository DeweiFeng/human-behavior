from dataloaders.meld_loader import MELDDataset
from dataloaders.mosei_loader import MOSEIDataset
from dataloaders.move4as_loader import Move4ASDataset
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
        **kwargs
    ) -> DataLoader:
        
        dataset = MOSEIDataset(data_dir, split, modalities)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=mosei_collate_fn,
            **kwargs
        )
    
    # Check available splits and modalities
    print("Available splits:", MOSEIDataset.get_available_splits(data_dir))
    print("Available modalities:", MOSEIDataset.get_available_modalities(data_dir))
    
    dataset = MOSEIDataset(data_dir, split=split, modalities=['vision', 'audio', 'text'])
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


def test_move4as_dataloader(data_dir, participants=None, tasks=None, modalities=None):
    def move4as_collate_fn(batch):
        """
        Custom collate function for Move4AS dataset to handle mixed data types.
        Time series data has variable lengths and mixed tensor/metadata structure.
        """
        # Separate tensors and non-tensors by modality
        collated = {}
        
        # Handle metadata (always present)
        for key in ['participant', 'task', 'session', 'group']:
            collated[key] = [item[key] for item in batch]
        
        # Handle each modality
        for modality in ['eeg', 'imu', 'mdata']:
            if modality in batch[0] and batch[0][modality] is not None:
                modality_data = {}
                
                if modality in ['eeg', 'imu']:
                    # Stack time series data - pad if needed for different lengths
                    data_tensors = [item[modality]['data'] for item in batch]
                    max_len = max(t.shape[0] for t in data_tensors)
                    
                    # Pad sequences to same length
                    padded_tensors = []
                    for tensor in data_tensors:
                        if tensor.shape[0] < max_len:
                            pad_size = max_len - tensor.shape[0]
                            padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size))
                            padded_tensors.append(padded)
                        else:
                            padded_tensors.append(tensor)
                    
                    modality_data['data'] = torch.stack(padded_tensors)
                    modality_data['original_lengths'] = [t.shape[0] for t in data_tensors]
                    
                    # Copy metadata
                    for key in ['channels', 'sensors', 'sampling_rate', 'shape_info']:
                        if key in batch[0][modality]:
                            modality_data[key] = batch[0][modality][key]
                
                elif modality == 'mdata':
                    # Handle motion capture data - more complex structure
                    for key in batch[0][modality].keys():
                        if key.endswith('_data'):
                            # Stack tensor data
                            tensors = [item[modality][key] for item in batch if key in item[modality]]
                            if tensors:
                                max_len = max(t.shape[0] for t in tensors)
                                padded_tensors = []
                                for tensor in tensors:
                                    if tensor.shape[0] < max_len:
                                        pad_size = max_len - tensor.shape[0]
                                        pad_dims = (0, 0) * (len(tensor.shape) - 1) + (0, pad_size)
                                        padded = torch.nn.functional.pad(tensor, pad_dims)
                                        padded_tensors.append(padded)
                                    else:
                                        padded_tensors.append(tensor)
                                modality_data[key] = torch.stack(padded_tensors)
                        else:
                            # Handle metadata
                            modality_data[key] = [item[modality].get(key) for item in batch]
                
                collated[modality] = modality_data
        
        return collated

    def create_move4as_dataloader(
        data_dir: str,
        participants: List[str] = None,
        tasks: List[str] = None,
        modalities: List[str] = None,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        
        dataset = Move4ASDataset(data_dir, participants, tasks, modalities)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=move4as_collate_fn,
            **kwargs
        )
    
    # Check available participants and tasks
    print("Available participants:", Move4ASDataset.get_available_participants(data_dir))
    print("Available tasks:", Move4ASDataset.get_available_tasks(data_dir))
    # Create dataset with specified filters
    dataset = Move4ASDataset(
        data_dir, 
        participants=participants, 
        tasks=tasks, 
        modalities=modalities or ['eeg', 'imu', 'mdata']
    )
    
    print("Trigger codes:", dataset.get_trigger_info())
    
    dataloader = create_move4as_dataloader(
        data_dir, 
        participants=participants, 
        tasks=tasks, 
        modalities=modalities,
        batch_size=2  # Small batch for testing
    )

    # Get sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample from {sample['participant']} {sample['task']} session {sample['session']}:")
        print("Sample keys:", list(sample.keys()))
        
        for key, value in sample.items():
            if key in ['eeg', 'imu']:
                if value is not None:
                    print(f"{key}:")
                    print(f"  data shape: {value['data'].shape} ({value['data'].dtype})")
                    print(f"  sampling rate: {value['sampling_rate']} Hz")
                    print(f"  duration: {value['data'].shape[0] / value['sampling_rate']:.2f} seconds")
            elif key == 'mdata' and value is not None:
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"  {subkey}: {subvalue.shape} ({subvalue.dtype})")
                    else:
                        print(f"  {subkey}: {subvalue}")
            elif not isinstance(value, dict):
                print(f"{key}: {value}")
        
        # Test session info
        info = dataset.get_session_info(0)
        print(f"\nSession info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Test batch
    print(f"\nTesting dataloader with batch_size=2:")
    for batch in dataloader:
        print("Batch contents:")
        for key, value in batch.items():
            if key in ['eeg', 'imu']:
                if 'data' in value:
                    print(f"  {key} data: {value['data'].shape}")
                    print(f"  {key} original_lengths: {value['original_lengths']}")
            elif key == 'mdata':
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"  {key}.{subkey}: {subvalue.shape}")
            elif isinstance(value, list):
                print(f"  {key}: {value}")
        break


if __name__ == "__main__":
    # test_meld_dataloader('/orcd/pool/003/dewei/dataset/meld/MELD.Raw', "test")
    # test_mosei_dataloader("datasets/MOSEI", "train")
    test_move4as_dataloader("datasets/move4as", participants=['P1', 'S1'], tasks=['dance'], modalities=['eeg', 'imu'])