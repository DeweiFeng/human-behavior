# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a human behavior analysis research project focused on multimodal datasets for autism detection and behavioral analysis. The project combines several datasets (MELD, MOSEI, Move4AS) and implements vision-language models for inference.

## Dataset Structure

- **MELD Dataset**: Emotion recognition from video data stored in tar.gz files with CSV labels
- **MOSEI Dataset**: Multimodal sentiment analysis with vision, audio, and text features
- **Move4AS Dataset**: 3D motion capture and EEG data for autism research, containing data from 20 neurotypical and 14 autistic participants performing walking and dancing tasks

## Key Components

### Data Loaders (`dataloaders/`)
- `meld_loader.py`: Handles MELD dataset with emotion classification (7 emotions: neutral, surprise, fear, sadness, joy, disgust, anger)
- `mosei_loader.py`: Processes MOSEI multimodal data with vision (50x35), audio (50x74), and text (50x300) features
- `move4as_loader.py`: Time series dataloader for Move4AS multimodal autism research data (EEG, IMU, motion capture)

### Loss Functions (`loss/`)
- `multi_task_loss.py`: Implements multi-task learning with CCC loss for affective tasks and various losses for pathology, social, cognitive, and personality predictions

### Autism Analysis (`autism/`)
- `inference/qwen_vl.py`: Qwen-VL vision-language model inference pipeline
- `inference/config_inference.yaml`: Configuration for video/frame analysis with ADOS scoring
- Video processing utilities for autism behavior analysis

## Common Development Tasks

### Running Tests
```bash
python dataloader_test.py
```

### Dataset Testing
- MELD: `test_meld_dataloader('/path/to/MELD.Raw', "test")`
- MOSEI: `test_mosei_dataloader("datasets/MOSEI", "train")`
- Move4AS: `test_move4as_dataloader("datasets/move4as", participants=['P1', 'S1'], tasks=['dance'], modalities=['eeg', 'imu'])`

### Inference Pipeline
The Qwen-VL model can process both video files and frame sequences for autism behavior analysis using the configuration in `autism/inference/config_inference.yaml`.

## Dependencies

Core dependencies from `requirements.txt`:
- torch==2.7.1
- pdf2image==1.17.0
- Pillow==11.2.1
- pytesseract==0.3.13

Additional dependencies likely needed:
- transformers (for Qwen-VL model)
- cv2 (OpenCV for video processing)
- scipy (for .mat file handling in Move4AS)

## Move4AS Dataloader Usage

The Move4AS dataset contains multimodal time series data for autism research. The dataloader handles EEG, IMU, and motion capture data with proper synchronization and batching.

### Basic Usage

```python
from dataloaders.move4as_loader import Move4ASDataset
from torch.utils.data import DataLoader

# Create dataset with filtering options
dataset = Move4ASDataset(
    data_dir="datasets/move4as",
    participants=['P1', 'P2', 'S1', 'S2'],  # P=clinical, S=control
    tasks=['dance', 'walk'],                 # motor imitation tasks
    modalities=['eeg', 'imu', 'mdata'],     # data types to load
    participant_groups=['clinical', 'control']  # alternative to participants list
)

# Access a sample
sample = dataset[0]
print(f"Participant: {sample['participant']}, Task: {sample['task']}")
print(f"EEG shape: {sample['eeg']['data'].shape}")  # (time_samples, 17_channels)
print(f"IMU shape: {sample['imu']['data'].shape}")  # (time_samples, 5_sensors)
```

### Data Structure

**EEG Data** (`eeg`):
- Shape: `(time_samples, 17_channels)` at 250 Hz
- Duration: ~360 seconds (90,000 samples)
- Raw EEG values in microvolts (μV)

**IMU Data** (`imu`):
- Shape: `(time_samples, 5_sensors)` at 250 Hz  
- Sensors: accelerometer, gyroscope, status flags
- Synchronized with EEG data

**Motion Capture Data** (`mdata`):
- `marker_data`: `(time_frames, 37_markers, 3_coords)` at ~60 Hz
- `rigidbody_data`: `(time_frames, 21_bodies, 7_params)` - position + quaternion
- `timestamp`: Time information for synchronization
- `cueRegister`, `stimRegister`: Experimental trigger markers

### Dataloader with Custom Collate

```python
from dataloader_test import test_move4as_dataloader

# Use built-in test function with custom collate for batching
test_move4as_dataloader(
    "datasets/move4as", 
    participants=['P1', 'S1'], 
    tasks=['dance'], 
    modalities=['eeg', 'imu']
)

# Custom dataloader creation
def create_move4as_dataloader(dataset, batch_size=4):
    from dataloader_test import move4as_collate_fn  # Import custom collate
    return DataLoader(dataset, batch_size=batch_size, collate_fn=move4as_collate_fn)
```

### Utility Methods

```python
# Check available data
participants = Move4ASDataset.get_available_participants("datasets/move4as")
# Returns: {'clinical': ['P1', 'P2', ...], 'control': ['S1', 'S2', ...]}

tasks = Move4ASDataset.get_available_tasks("datasets/move4as")
# Returns: ['dance', 'walk']

# Get session information
info = dataset.get_session_info(0)
print(f"Duration: {info['eeg_duration_sec']} seconds")
print(f"Available modalities: {info['available_modalities']}")

# Get trigger codes for experimental design
triggers = dataset.get_trigger_info()
# Returns: {1: 'baseline_fixation', 2: 'instruction', ...}
```

### Time Series Model Integration

The dataloader is designed for compatibility with popular time series models:

```python
# For RNNs/LSTMs
eeg_data = sample['eeg']['data']  # (seq_len, features)
lstm_input = eeg_data.unsqueeze(0)  # (batch, seq_len, features)

# For Transformers  
transformer_input = eeg_data.transpose(0, 1)  # (features, seq_len)

# For CNNs (temporal convolution)
cnn_input = eeg_data.transpose(0, 1).unsqueeze(0)  # (batch, channels, time)

# Handling variable lengths in batch
batch = next(iter(dataloader))
original_lengths = batch['eeg']['original_lengths']  # For masking
padded_data = batch['eeg']['data']  # Pre-padded sequences
```

### Data Alignment and Synchronization

EEG and motion capture data use trigger codes for temporal alignment:
- **Phase triggers**: 1=Baseline, 2=Instruction, 3=High-tone beep, 4=Execution, 5=Low-tone beep, 6=Return
- **Task codes**: 1=Confident walk/Solo dancing, 2=Walk naturally/Body shake, 3=Sad walk/Dancing with pair
- **Sampling rates**: EEG/IMU at 250 Hz, motion capture at ~60 Hz
- **Duration**: All modalities span ~360 seconds per session