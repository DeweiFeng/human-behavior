#!/usr/bin/env python3
"""
Simple example of how to use the MOSEI training function.

This script demonstrates the basic usage of the mosei_train function
for training a multimodal sentiment analysis model on the MOSEI dataset.
"""

from dataloaders.mosei_loader import mosei_training, mosei_process_batch, mosei_evaluate


def main():
    """Main function demonstrating MOSEI training setup."""
    
    # Configuration for training
    config = {
        'data_dir': '/path/to/mosei/data',  # Update this to your MOSEI data path
        'save_path': 'checkpoints/mosei_model.pt',
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'train_split': 'train',
        'val_split': 'valid',
        'modalities': ['vision', 'audio', 'text'],  # Use all modalities
        'device': None,  # Auto-detect GPU/CPU
        'num_workers': 4,
        'verbose': True
    }
    
    print("MOSEI Training Setup Example")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Setup training
    print("Setting up MOSEI training...")
    try:
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
        
        # Example of processing a batch
        print("\nExample batch processing:")
        for batch in train_loader:
            processed_batch = mosei_process_batch(batch, training_config['device'])
            print(f"  Batch keys: {list(processed_batch.keys())}")
            for key, value in processed_batch.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: {value.shape}")
            break
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the data directory path is correct.")
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 