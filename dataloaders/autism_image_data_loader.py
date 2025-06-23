import os
from typing import Dict, List, Tuple, Any

from torch.utils.data import Dataset


class AutismImageDataset(Dataset):
    """
    Autism Image Dataset Loader for Kaggle dataset:
    https://www.kaggle.com/datasets/cihan063/autism-image-data
    
    This dataset contains facial images for autism detection research.
    Classes: 'Autistic' (label=1) and 'Non_Autistic' (label=0)
    
    **Dataset Structure (after download to datasets/autism_images/):**
    ```
    datasets/autism_images/
    └── AutismDataset/
        ├── train/                    # 2540 images total
        │   ├── Autistic.0.jpg        # 1270 autistic images (naming: Autistic.{number}.jpg)
        │   ├── Autistic.1.jpg
        │   ├── ...
        │   ├── Non_Autistic.0.jpg    # 1270 non-autistic images (naming: Non_Autistic.{number}.jpg)
        │   ├── Non_Autistic.1.jpg
        │   └── ...
        ├── test/                     # 300 images total
        │   ├── Autistic.0.jpg        # 150 autistic images (flat structure, same naming)
        │   ├── ...
        │   ├── Non_Autistic.0.jpg    # 150 non-autistic images
        │   └── ...
        ├── valid/                    # 100 images total
        │   ├── Autistic/             # 50 autistic images (subdirectory structure)
        │   │   ├── 01.jpg
        │   │   ├── 02.jpg
        │   │   └── ...
        │   └── Non_Autistic/         # 50 non-autistic images
        │       ├── 01.jpg
        │       ├── 02.jpg
        │       └── ...
        └── consolidated/             # Additional folder (not used by default)
            ├── Autistic/
            └── Non_Autistic/
    ```
    
    **Dataset Statistics:**
    - Total: 2940 images across all splits
    - Train: 2540 images (1270 autistic, 1270 non-autistic) - perfectly balanced
    - Test: 300 images (150 autistic, 150 non-autistic) - perfectly balanced  
    - Valid: 100 images (50 autistic, 50 non-autistic) - perfectly balanced
    
    **Note:** The dataset has inconsistent structure:
    - Train/Test splits: Flat directory with filename-based class labels
    - Valid split: Subdirectory-based class organization
    - All splits are perfectly balanced (50% autistic, 50% non-autistic)
    
    **Usage:**
    ```python
    # Download dataset
    AutismImageDataset.download_dataset("datasets/autism_images")
    
    # Load different splits
    train_dataset = AutismImageDataset("datasets/autism_images", split="train")
    valid_dataset = AutismImageDataset("datasets/autism_images", split="valid") 
    test_dataset = AutismImageDataset("datasets/autism_images", split="test")
    ```
    """
    
    CLASS_MAP = {
        'Autistic': 1,
        'Non_Autistic': 0
    }
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        """
        Initialize Autism Image dataset.
        
        Args:
            data_dir: Directory containing the autism image data
            split: Dataset split ('train', 'test', or 'valid')
            transform: Optional image transforms
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Validate split
        if split not in ['train', 'test', 'valid']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'test', or 'valid'")
        
        # Build file index
        self.samples = self._build_file_index()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{split}' in {data_dir}")
        
        print(f"Autism Image {split} dataset loaded:")
        print(f"  Number of samples: {len(self.samples)}")
        
        # Print class distribution
        class_counts = {}
        for _, label in self.samples:
            class_name = self._get_class_name(label)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
    
    def _build_file_index(self) -> List[Tuple[str, int]]:
        """Build index of image files and labels."""
        samples = []
        
        # Dataset is downloaded to AutismDataset subfolder
        actual_split_dir = os.path.join(self.data_dir, "AutismDataset", self.split)
        
        if not os.path.exists(actual_split_dir):
            raise FileNotFoundError(f"Split directory not found: {actual_split_dir}")
        
        if self.split == 'valid':
            # Valid split has subdirectory structure: valid/Autistic/, valid/Non_Autistic/
            for class_name in os.listdir(actual_split_dir):
                class_dir = os.path.join(actual_split_dir, class_name)
                
                if not os.path.isdir(class_dir):
                    continue
                
                if class_name not in self.CLASS_MAP:
                    print(f"Warning: Unknown class '{class_name}' found in {class_dir}")
                    continue
                
                label = self.CLASS_MAP[class_name]
                
                # Find all image files in class directory
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(class_dir, filename)
                        samples.append((file_path, label))
        
        else:
            # Train and test splits have flat structure with filename-based labels
            # Files are named like: Autistic.0.jpg, Non_Autistic.1.jpg, etc.
            for filename in os.listdir(actual_split_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Extract class name from filename
                for class_name in self.CLASS_MAP.keys():
                    if filename.startswith(f"{class_name}."):
                        label = self.CLASS_MAP[class_name]
                        file_path = os.path.join(actual_split_dir, filename)
                        samples.append((file_path, label))
                        break
                else:
                    print(f"Warning: Cannot determine class for file: {filename}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and label
        """
        file_path, label = self.samples[idx]
        
        # Load image
        try:
            from PIL import Image
            image = Image.open(file_path).convert('RGB')
        except ImportError:
            raise ImportError("PIL/Pillow is required for image loading. Install with: pip install Pillow")
        except Exception as e:
            raise RuntimeError(f"Error loading image {file_path}: {e}")
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'label': label,
            'file_path': file_path,
            'class_name': self._get_class_name(label)
        }
        
        return sample
    
    def _get_class_name(self, label: int) -> str:
        """Get class name from label."""
        for class_name, class_label in self.CLASS_MAP.items():
            if class_label == label:
                return class_name
        return "unknown"
    
    @classmethod
    def get_class_names(cls) -> List[str]:
        """Get list of class names."""
        return list(cls.CLASS_MAP.keys())
    
    @classmethod
    def download_dataset(cls, data_dir: str, force_download: bool = False) -> bool:
        """
        Download the Autism Image dataset from Kaggle.
        
        Args:
            data_dir: Directory to download and extract the dataset
            force_download: If True, re-download even if data exists
            
        Returns:
            True if download successful, False otherwise
        """
        # Check if dataset already exists
        if not force_download and cls._check_dataset_exists(data_dir):
            print(f"Dataset already exists at {data_dir}")
            return True
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            # Import kaggle API
            try:
                import kaggle
            except ImportError:
                raise ImportError(
                    "Kaggle API is required for dataset download. Install with: pip install kaggle\n"
                    "Also ensure you have configured your Kaggle API credentials."
                )
            
            print("Downloading Autism Image dataset from Kaggle...")
            
            # Download dataset using Kaggle API
            kaggle.api.dataset_download_files(
                'cihan063/autism-image-data',
                path=data_dir,
                unzip=True
            )
            
            print(f"Dataset downloaded and extracted to {data_dir}")
            
            # Verify the download
            if cls._check_dataset_exists(data_dir):
                print("Dataset download completed successfully!")
                return True
            else:
                print("Warning: Dataset structure verification failed after download")
                return False
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nAlternative download instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/cihan063/autism-image-data")
            print("2. Download the dataset manually")
            print(f"3. Extract to: {data_dir}")
            print("4. Ensure the directory structure matches: data_dir/[train|test|valid]/[Autistic|Non_Autistic]/")
            return False
    
    @classmethod
    def _check_dataset_exists(cls, data_dir: str) -> bool:
        """Check if the dataset exists and has the expected structure."""
        autism_dataset_dir = os.path.join(data_dir, "AutismDataset")
        if not os.path.exists(autism_dataset_dir):
            return False
        
        required_splits = ['train', 'test', 'valid']
        
        for split in required_splits:
            split_dir = os.path.join(autism_dataset_dir, split)
            if not os.path.exists(split_dir):
                return False
            
            if split == 'valid':
                # Valid has subdirectory structure
                for class_name in ['Autistic', 'Non_Autistic']:
                    class_dir = os.path.join(split_dir, class_name)
                    if not os.path.exists(class_dir):
                        return False
                    
                    # Check if class directory has any image files
                    has_images = any(
                        f.lower().endswith(('.png', '.jpg', '.jpeg'))
                        for f in os.listdir(class_dir)
                    )
                    if not has_images:
                        return False
            else:
                # Train and test have flat structure with filename patterns
                files = os.listdir(split_dir)
                has_autistic = any(f.startswith('Autistic.') and f.endswith('.jpg') for f in files)
                has_non_autistic = any(f.startswith('Non_Autistic.') and f.endswith('.jpg') for f in files)
                
                if not (has_autistic and has_non_autistic):
                    return False
        
        return True
    
    @classmethod
    def get_dataset_info(cls, data_dir: str) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = {
            'exists': cls._check_dataset_exists(data_dir),
            'splits': {},
            'total_samples': 0
        }
        
        if not info['exists']:
            return info
        
        autism_dataset_dir = os.path.join(data_dir, "AutismDataset")
        splits = ['train', 'test', 'valid']
        
        for split in splits:
            split_info = {'classes': {}, 'total': 0}
            split_dir = os.path.join(autism_dataset_dir, split)
            
            if os.path.exists(split_dir):
                if split == 'valid':
                    # Valid has subdirectory structure
                    for class_name in cls.CLASS_MAP.keys():
                        class_dir = os.path.join(split_dir, class_name)
                        if os.path.exists(class_dir):
                            count = len([
                                f for f in os.listdir(class_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                            ])
                            split_info['classes'][class_name] = count
                            split_info['total'] += count
                else:
                    # Train and test have flat structure with filename patterns
                    files = os.listdir(split_dir)
                    for class_name in cls.CLASS_MAP.keys():
                        count = len([
                            f for f in files
                            if f.startswith(f"{class_name}.") and f.lower().endswith('.jpg')
                        ])
                        split_info['classes'][class_name] = count
                        split_info['total'] += count
                
                info['splits'][split] = split_info
                info['total_samples'] += split_info['total']
        
        return info


if __name__ == "__main__":
    # Example usage
    pass