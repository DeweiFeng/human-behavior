#!/usr/bin/env python3
"""
Example of how to integrate MOSEI training with the existing train_script.py structure.

This script shows how to use the mosei_training function to set up MOSEI-specific
data loaders and configurations that can work with the existing MultiTaskMentalHealthModel.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor
from models import MultiTaskMentalHealthModel
from tqdm import tqdm
import numpy as np
import os

# Import MOSEI training functions
from dataloaders.mosei_loader import mosei_training, mosei_process_batch, mosei_evaluate


def main():
    """Main function demonstrating MOSEI integration with train_script.py."""
    
    # === Config ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEXT_MODEL = "meta-llama/Llama-3.2-1B"
    AUDIO_MODEL = "microsoft/wavlm-base-plus"
    BATCH_SIZE = 32
    EPOCHS = 5
    SAVE_PATH = "checkpoints/mosei_multitask_model.pt"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # MOSEI-specific configuration
    MOSEI_DATA_DIR = "/path/to/mosei/data"  # Update this path
    
    print("MOSEI Training Integration Example")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Text model: {TEXT_MODEL}")
    print(f"Audio model: {AUDIO_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # === Setup MOSEI Training ===
    print("Setting up MOSEI training...")
    try:
        mosei_setup = mosei_training(
            data_dir=MOSEI_DATA_DIR,
            train_split="train",
            val_split="valid",
            modalities=["vision", "audio", "text"],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=1e-5,
            weight_decay=1e-5,
            device=DEVICE,
            num_workers=4,
            save_path=SAVE_PATH,
            verbose=True
        )
        
        # Extract components
        train_loader = mosei_setup['train_loader']
        val_loader = mosei_setup['val_loader']
        dataset_info = mosei_setup['dataset_info']
        training_config = mosei_setup['training_config']
        
        print(f"MOSEI setup completed!")
        print(f"Dataset: {dataset_info['name']}")
        print(f"Train samples: {dataset_info['train_samples']}")
        print(f"Val samples: {dataset_info['val_samples']}")
        print(f"Task type: {dataset_info['task_type']}")
        print()
        
    except Exception as e:
        print(f"Failed to setup MOSEI training: {e}")
        return
    
    # === Tokenizers ===
    print("Loading tokenizers...")
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    text_tokenizer.pad_token = text_tokenizer.eos_token
    audio_tokenizer = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
    
    # === Model, Optimizer, Loss ===
    print("Initializing model...")
    model = MultiTaskMentalHealthModel(text_model=TEXT_MODEL, audio_model=AUDIO_MODEL).to(DEVICE)
    
    # Use MOSEI-specific optimizer and loss
    optimizer = training_config['optimizer_class'](
        model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config['weight_decay']
    )
    criterion = training_config['criterion']  # MSE loss for regression
    
    # Learning rate scheduler
    scheduler = training_config['scheduler_class'](
        optimizer, **training_config['scheduler_kwargs']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # === Training Loop ===
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_mae = 0.0
            
            # Process MOSEI batch
            processed_inputs = mosei_process_batch(batch, DEVICE)
            labels = processed_inputs['labels']
            
            # Prepare inputs for MultiTaskMentalHealthModel
            # Note: This is a simplified example - you may need to adapt this
            # based on how your model expects the inputs
            
            # For demonstration, we'll create dummy inputs
            # In practice, you'd need to convert MOSEI features to the format expected by your model
            batch_size = labels.shape[0]
            
            # Create dummy text inputs (you'd need to convert MOSEI text features)
            text_inputs = {
                'input_ids': torch.randint(0, 1000, (batch_size, 50)).to(DEVICE),
                'attention_mask': torch.ones(batch_size, 50).to(DEVICE)
            }
            
            # Create dummy audio inputs (you'd need to convert MOSEI audio features)
            audio_inputs = torch.randn(batch_size, 16000).to(DEVICE)  # 1 second of audio
            
            # Create dummy face and gesture inputs
            face_input = torch.randn(batch_size, 35).to(DEVICE)
            pose_input = torch.randn(batch_size, 6).to(DEVICE)
            
            # Forward pass
            outputs = model(
                text_inputs, audio_inputs, face_input, pose_input, 
                task="mosei_sentiment"  # You'd need to add this task to your model
            )
            
            # Calculate loss (assuming outputs are regression values)
            loss = criterion(outputs.squeeze(), labels)
            mae = torch.mean(torch.abs(outputs.squeeze() - labels))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if training_config['gradient_clip_val'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip_val'])
            
            optimizer.step()
            
            batch_loss += loss.item()
            batch_mae += mae.item()
            num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae.item():.4f}'
            })
        
        avg_train_loss = total_loss / num_batches
        avg_train_mae = total_mae / num_batches
        
        # Validation phase
        print(f"\nValidating...")
        val_metrics = mosei_evaluate(model, val_loader, DEVICE, criterion)
        val_loss = val_metrics['val_loss']
        val_mae = val_metrics['val_mae']
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  Saved best model (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= training_config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main() 