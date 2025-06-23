import os
import glob
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import scipy.io
except ImportError:
    raise ImportError("scipy is required for loading .mat files. Install with: pip install scipy")


class Move4ASDataset(Dataset):
    """
    Move4AS (Movement Imitation for Autism Spectrum) Dataset Loader.
    
    The Move4AS dataset contains multimodal time series data with:
    - EEG: 17 channels at 250 Hz (neural activity)
    - IMU: 5 sensors at 250 Hz (inertial measurement)
    - Motion: 37 markers + 21 rigid bodies at ~50 Hz (3D motion capture)
    
    Participants: P1-P14 (autism group), S1-S20 (control group)
    Tasks: walking and dancing motor imitation
    """
    
    PARTICIPANT_GROUPS = {
        'P': 'clinical',    # Autism group
        'S': 'control'      # Neurotypical group
    }
    
    TASKS = ['dance', 'walk']
    MODALITIES = ['eeg', 'imu', 'mdata']
    
    # Data alignment triggers (from README)
    TRIGGER_CODES = {
        1: "baseline_fixation",
        2: "instruction", 
        3: "high_tone_beep",
        4: "execution",
        5: "low_tone_beep",
        6: "return_position"
    }
    
    def __init__(self, 
                 data_dir: str, 
                 participants: Optional[List[str]] = None,
                 tasks: Optional[List[str]] = None,
                 modalities: Optional[List[str]] = None,
                 participant_groups: Optional[List[str]] = None):
        """
        Initialize Move4AS dataset.
        
        Args:
            data_dir: Directory containing Move4AS Dataset folder
            participants: List of participant IDs (e.g., ['P1', 'S1']) or None for all
            tasks: List of tasks (['dance', 'walk']) or None for all
            modalities: List of modalities (['eeg', 'imu', 'mdata']) or None for all
            participant_groups: List of groups (['clinical', 'control']) or None for all
        """
        self.data_dir = os.path.join(data_dir, "Dataset")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Move4AS Dataset directory not found at {self.data_dir}")
        
        # Set defaults
        if modalities is None:
            modalities = self.MODALITIES.copy()
        if tasks is None:
            tasks = self.TASKS.copy()
            
        self.modalities = modalities
        self.tasks = tasks
        
        # Build file index
        self.file_index = self._build_file_index(participants, tasks, participant_groups)
        
        if len(self.file_index) == 0:
            raise ValueError("No valid files found with the specified criteria")
        
        print(f"Move4AS dataset loaded:")
        print(f"  Number of sessions: {len(self.file_index)}")
        print(f"  Modalities: {self.modalities}")
        print(f"  Tasks: {self.tasks}")
        
        # Print participant distribution
        groups = {}
        for item in self.file_index:
            group = self.PARTICIPANT_GROUPS.get(item['participant'][0], 'unknown')
            groups[group] = groups.get(group, set())
            groups[group].add(item['participant'])
        
        for group, participants in groups.items():
            print(f"  {group.capitalize()} participants: {len(participants)} ({sorted(participants)})")

    def _build_file_index(self, 
                         participants: Optional[List[str]], 
                         tasks: List[str],
                         participant_groups: Optional[List[str]]) -> List[Dict]:
        """Build index of available files."""
        file_index = []
        
        # Get all participant directories
        participant_dirs = glob.glob(os.path.join(self.data_dir, "[PS]*"))
        
        for participant_dir in participant_dirs:
            participant_id = os.path.basename(participant_dir)
            
            # Filter by participant list
            if participants is not None and participant_id not in participants:
                continue
                
            # Filter by participant group
            if participant_groups is not None:
                group = self.PARTICIPANT_GROUPS.get(participant_id[0])
                if group not in participant_groups:
                    continue
            
            # Check each task
            for task in tasks:
                task_dir = os.path.join(participant_dir, task)
                if not os.path.exists(task_dir):
                    continue
                
                # Find all sessions for this participant/task
                session_files = {}
                for modality in self.modalities:
                    pattern = os.path.join(task_dir, f"{modality}_{participant_id}{task}*.mat")
                    files = glob.glob(pattern)
                    session_files[modality] = files
                
                # Group by session number
                sessions = set()
                for files in session_files.values():
                    for file_path in files:
                        # Extract session number from filename
                        basename = os.path.basename(file_path)
                        # e.g., "eeg_P1dance1.mat" -> extract "1"
                        session_num = basename.split(task)[-1].replace('.mat', '')
                        if session_num.isdigit():
                            sessions.add(session_num)
                
                # Create index entry for each session
                for session_num in sessions:
                    entry = {
                        'participant': participant_id,
                        'task': task,
                        'session': session_num,
                        'group': self.PARTICIPANT_GROUPS.get(participant_id[0], 'unknown'),
                        'files': {}
                    }
                    
                    # Check which modalities have files for this session
                    for modality in self.modalities:
                        file_path = os.path.join(task_dir, 
                                               f"{modality}_{participant_id}{task}{session_num}.mat")
                        if os.path.exists(file_path):
                            entry['files'][modality] = file_path
                    
                    # Only add if at least one modality file exists
                    if entry['files']:
                        file_index.append(entry)
        
        return file_index

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single session sample.
        
        Returns:
            Dictionary containing time series data and metadata
        """
        entry = self.file_index[idx]
        sample = {
            'participant': entry['participant'],
            'task': entry['task'], 
            'session': entry['session'],
            'group': entry['group']
        }
        
        # Load each available modality
        for modality in self.modalities:
            if modality in entry['files']:
                try:
                    data = self._load_modality_data(entry['files'][modality], modality)
                    sample[modality] = data
                except Exception as e:
                    print(f"Warning: Failed to load {modality} for {entry['participant']} {entry['task']} {entry['session']}: {e}")
                    sample[modality] = None
        
        return sample

    def _load_modality_data(self, file_path: str, modality: str) -> Dict[str, torch.Tensor]:
        """Load and process data for a specific modality."""
        mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        if modality == 'eeg':
            # EEG data: (17 channels, N_samples) at 250 Hz
            eeg_data = mat_data['eegDataT']
            return {
                'data': torch.tensor(eeg_data.T, dtype=torch.float32),  # (N_samples, 17)
                'channels': 17,
                'sampling_rate': 250,
                'shape_info': 'time_samples x channels'
            }
            
        elif modality == 'imu':
            # IMU data: (5 sensors, N_samples) at 250 Hz  
            imu_data = mat_data['imuDataT']
            return {
                'data': torch.tensor(imu_data.T, dtype=torch.float32),  # (N_samples, 5)
                'sensors': 5,
                'sampling_rate': 250,
                'shape_info': 'time_samples x sensors'
            }
            
        elif modality == 'mdata':
            # Motion capture data: complex structure at ~50 Hz
            result = {}
            
            # Extract marker data if available
            if 'markerData' in mat_data:
                marker_data = mat_data['markerData']
                if hasattr(marker_data, '__len__') and len(marker_data) > 0:
                    # Extract 3D coordinates for all markers across time
                    n_frames = len(marker_data)
                    marker_coords = np.zeros((n_frames, 37, 3))  # 37 markers, 3D coords
                    
                    for i, frame in enumerate(marker_data):
                        if hasattr(frame, 'shape') and frame.shape == (3, 37):
                            marker_coords[i] = frame.T  # (37, 3)
                    
                    result['marker_data'] = torch.tensor(marker_coords, dtype=torch.float32)
                    result['marker_shape_info'] = 'time_frames x markers x coordinates(xyz)'
            
            # Extract rigid body data if available
            if 'rigidbodyData' in mat_data:
                rb_data = mat_data['rigidbodyData']
                if hasattr(rb_data, '__len__') and len(rb_data) > 0:
                    n_frames = len(rb_data)
                    rb_coords = np.zeros((n_frames, 21, 7))  # 21 bodies, 7 params each
                    
                    for i, frame in enumerate(rb_data):
                        if hasattr(frame, 'shape') and frame.shape == (7, 21):
                            rb_coords[i] = frame.T  # (21, 7)
                    
                    result['rigidbody_data'] = torch.tensor(rb_coords, dtype=torch.float32)
                    result['rigidbody_shape_info'] = 'time_frames x bodies x params(x,y,z,qx,qy,qz,qw)'
            
            # Extract timing information
            for key in ['timestamp', 'startFrame', 'frameCount']:
                if key in mat_data:
                    result[key] = torch.tensor(mat_data[key], dtype=torch.float32)
            
            # Extract trigger information
            for key in ['cueRegister', 'stimRegister']:
                if key in mat_data:
                    result[key] = torch.tensor(mat_data[key], dtype=torch.float32)
            
            # Calculate sampling rate from timestamps if available
            if 'timestamp' in result and len(result['timestamp']) > 1:
                dt = result['timestamp'][1] - result['timestamp'][0]
                sampling_rate = 1.0 / dt.item() if dt.item() > 0 else 50.0
                result['sampling_rate'] = sampling_rate
            else:
                result['sampling_rate'] = 50.0  # Default estimate
                
            return result
        
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def get_session_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific session."""
        entry = self.file_index[idx]
        sample = self[idx]
        
        info = {
            'participant': entry['participant'],
            'task': entry['task'],
            'session': entry['session'], 
            'group': entry['group'],
            'available_modalities': list(entry['files'].keys())
        }
        
        # Add data shape information
        for modality in self.modalities:
            if modality in sample and sample[modality] is not None:
                if modality in ['eeg', 'imu']:
                    data_shape = sample[modality]['data'].shape
                    sampling_rate = sample[modality]['sampling_rate']
                    duration = data_shape[0] / sampling_rate
                    info[f'{modality}_shape'] = data_shape
                    info[f'{modality}_duration_sec'] = duration
                elif modality == 'mdata':
                    if 'marker_data' in sample[modality]:
                        info['mdata_marker_shape'] = sample[modality]['marker_data'].shape
                    if 'rigidbody_data' in sample[modality]:
                        info['mdata_rigidbody_shape'] = sample[modality]['rigidbody_data'].shape
                    if 'sampling_rate' in sample[modality]:
                        info['mdata_sampling_rate'] = sample[modality]['sampling_rate']
        
        return info

    @classmethod 
    def get_available_participants(cls, data_dir: str) -> Dict[str, List[str]]:
        """Get available participants grouped by clinical status."""
        dataset_dir = os.path.join(data_dir, "Dataset")
        if not os.path.exists(dataset_dir):
            return {}
        
        participant_dirs = glob.glob(os.path.join(dataset_dir, "[PS]*"))
        participants = {'clinical': [], 'control': []}
        
        for participant_dir in participant_dirs:
            participant_id = os.path.basename(participant_dir)
            group = cls.PARTICIPANT_GROUPS.get(participant_id[0])
            if group in participants:
                participants[group].append(participant_id)
        
        return {k: sorted(v) for k, v in participants.items()}

    @classmethod
    def get_available_tasks(cls, data_dir: str, participant: str = None) -> List[str]:
        """Get available tasks for a participant or all participants."""
        dataset_dir = os.path.join(data_dir, "Dataset")
        if not os.path.exists(dataset_dir):
            return []
        
        if participant:
            participant_dir = os.path.join(dataset_dir, participant)
            if os.path.exists(participant_dir):
                tasks = []
                for task in cls.TASKS:
                    if os.path.exists(os.path.join(participant_dir, task)):
                        tasks.append(task)
                return tasks
        else:
            # Return all tasks that exist for any participant
            all_tasks = set()
            participant_dirs = glob.glob(os.path.join(dataset_dir, "[PS]*"))
            for participant_dir in participant_dirs:
                for task in cls.TASKS:
                    if os.path.exists(os.path.join(participant_dir, task)):
                        all_tasks.add(task)
            return sorted(all_tasks)
        
        return []

    def get_trigger_info(self) -> Dict[int, str]:
        """Get mapping of trigger codes to their meanings."""
        return self.TRIGGER_CODES.copy()


if __name__ == "__main__":
    # Example usage
    pass