#!/usr/bin/env python3
"""
Test script for MOSEI training function.
This script demonstrates how to use the mosei_train function.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.mosei_loader import mosei_training, mosei_process_batch, mosei_evaluate, MOSEIDataset


def test_mosei_training():
    """Test the MOSEI training setup with a small dataset."""
    
    # Configuration
    data_dir = "path/to/your/mosei/data"  # Update this path
    save_path = "checkpoints/mosei_test.pt"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        print("Please update the data_dir path in this script.")
        return
    
    # Check available splits
    try:
        available_splits = MOSEIDataset.get_available_splits(data_dir)
        print(f"Available splits: {available_splits}")
        
        if not available_splits:
            print("No data splits found. Please check your data directory.")
            return
            
        # Use first available split for training if 'train' not available
        train_split = "train" if "train" in available_splits else available_splits[0]
        val_split = "valid" if "valid" in available_splits else available_splits[-1]
        
        print(f"Using train split: {train_split}")
        print(f"Using validation split: {val_split}")
        
    except Exception as e:
        print(f"Error checking data: {e}")
        return
    
    # Training configuration
    config = {
        'data_dir': data_dir,
        'save_path': save_path,
        'batch_size': 16,  # Smaller batch size for testing
        'epochs': 5,       # Fewer epochs for testing
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'train_split': train_split,
        'val_split': val_split,
        'modalities': ['vision', 'audio', 'text'],  # Use all modalities
        'device': None,  # Auto-detect
        'num_workers': 2,
        'verbose': True
    }
    
    print("Starting MOSEI training setup with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Setup training
        training_setup = mosei_training(**config)
        
        print("\nTraining setup completed successfully!")
        print(f"Train samples: {training_setup['dataset_info']['train_samples']}")
        print(f"Val samples: {training_setup['dataset_info']['val_samples']}")
        print(f"Modalities: {training_setup['dataset_info']['modalities']}")
        print(f"Task type: {training_setup['dataset_info']['task_type']}")
        
        # Access the data loaders
        train_loader = training_setup['train_loader']
        val_loader = training_setup['val_loader']
        dataset_info = training_setup['dataset_info']
        training_config = training_setup['training_config']
        
        print(f"\nData loaders ready:")
        print(f"  Train loader: {len(train_loader)} batches")
        print(f"  Val loader: {len(val_loader)} batches")
        print(f"  Loss function: {training_config['criterion']}")
        print(f"  Optimizer: {training_config['optimizer_class']}")
        
        # Test batch processing
        print("\nTesting batch processing:")
        for batch in train_loader:
            processed_batch = mosei_process_batch(batch, training_config['device'])
            print(f"  Batch keys: {list(processed_batch.keys())}")
            for key, value in processed_batch.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: {value.shape}")
            break
        
    except Exception as e:
        print(f"Training setup failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_mosei_dataset():
    """Test the MOSEI dataset loading."""
    
    data_dir = "path/to/your/mosei/data"  # Update this path
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return
    
    try:
        # Test dataset loading
        print("Testing MOSEI dataset loading...")
        
        # Check available splits
        available_splits = MOSEIDataset.get_available_splits(data_dir)
        print(f"Available splits: {available_splits}")
        
        if available_splits:
            split = available_splits[0]
            
            # Load dataset
            dataset = MOSEIDataset(data_dir, split=split, modalities=['vision', 'audio', 'text'])
            print(f"Dataset loaded successfully!")
            print(f"Number of samples: {len(dataset)}")
            print(f"Modality shapes: {dataset.get_modality_shapes()}")
            print(f"Label info: {dataset.get_label_info()}")
            
            # Test getting a sample
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        
    except Exception as e:
        print(f"Dataset test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("MOSEI Training Test Script")
    print("=" * 50)
    
    # Test dataset loading first
    print("\n1. Testing dataset loading...")
    test_mosei_dataset()
    
    # Test training
    print("\n2. Testing training...")
    test_mosei_training()
    
    print("\nTest script completed!") 