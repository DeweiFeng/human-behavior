#!/usr/bin/env python3
"""
Test script for Autism Image Dataset Loader.

This script tests the AutismImageDataset class functionality including:
- Dataset verification
- Dataset info retrieval
- Loading different splits (train, valid, test)
- Basic functionality testing

Usage:
    python tests/test_autism_image_dataloader.py
"""

import sys
import os

# Add parent directory to path to import dataloaders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.autism_image_data_loader import AutismImageDataset


def test_dataset_verification():
    """Test dataset structure verification."""
    print('=== Testing Dataset Verification ===')
    exists = AutismImageDataset._check_dataset_exists('datasets/autism_images')
    print(f'Dataset exists: {exists}')
    return exists


def test_dataset_info():
    """Test dataset info retrieval."""
    print('\n=== Getting Dataset Info ===')
    info = AutismImageDataset.get_dataset_info('datasets/autism_images')
    print(f'Dataset info: {info}')
    
    if info['exists']:
        print(f"Total samples: {info['total_samples']}")
        for split, split_info in info['splits'].items():
            print(f"{split}: {split_info['total']} samples")
            for class_name, count in split_info['classes'].items():
                print(f"  {class_name}: {count}")
    
    return info


def calculate_detailed_statistics():
    """Calculate and print detailed dataset statistics."""
    print('\n=== Detailed Dataset Statistics ===')
    
    try:
        # Get basic info using the built-in method
        info = AutismImageDataset.get_dataset_info('datasets/autism_images')
        
        if not info['exists']:
            print("Dataset not found!")
            return None
        
        # Calculate detailed statistics
        stats = {
            'total_samples': info['total_samples'],
            'splits': {}
        }
        
        for split in ['train', 'test', 'valid']:
            if split in info['splits']:
                split_info = info['splits'][split]
                autistic_count = split_info['classes'].get('Autistic', 0)
                non_autistic_count = split_info['classes'].get('Non_Autistic', 0)
                total_split = split_info['total']
                
                autistic_pct = (autistic_count / total_split * 100) if total_split > 0 else 0
                non_autistic_pct = (non_autistic_count / total_split * 100) if total_split > 0 else 0
                
                stats['splits'][split] = {
                    'total': total_split,
                    'autistic': autistic_count,
                    'non_autistic': non_autistic_count,
                    'autistic_percentage': autistic_pct,
                    'non_autistic_percentage': non_autistic_pct,
                    'balanced': abs(autistic_count - non_autistic_count) <= 1
                }
        
        # Print formatted statistics
        print("Dataset Statistics Summary:")
        print(f"Total samples across all splits: {stats['total_samples']}")
        print()
        
        for split, split_stats in stats['splits'].items():
            print(f"{split.upper()} Split:")
            print(f"  Total samples: {split_stats['total']}")
            print(f"  Autistic samples: {split_stats['autistic']} ({split_stats['autistic_percentage']:.1f}%)")
            print(f"  Non-Autistic samples: {split_stats['non_autistic']} ({split_stats['non_autistic_percentage']:.1f}%)")
            print(f"  Balanced: {'Yes' if split_stats['balanced'] else 'No'}")
            print()
        
        # Generate docstring statistics text
        print("=== Statistics for Docstring ===")
        docstring_stats = []
        for split, split_stats in stats['splits'].items():
            docstring_stats.append(f"    - {split}: {split_stats['total']} images ({split_stats['autistic']} autistic, {split_stats['non_autistic']} non-autistic)")
        
        docstring_text = "**Dataset Statistics:**\n" + "\n".join(docstring_stats)
        print(docstring_text)
        
        return stats
        
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return None


def test_train_split():
    """Test loading train split."""
    print('\n=== Loading Train Split ===')
    try:
        train_dataset = AutismImageDataset('datasets/autism_images', split='train')
        print(f'Train dataset length: {len(train_dataset)}')
        
        # Test first sample
        sample = train_dataset[0]
        print(f'First sample keys: {list(sample.keys())}')
        print(f'First sample label: {sample["label"]}')
        print(f'First sample class: {sample["class_name"]}')
        print(f'Image type: {type(sample["image"])}')
        
        return train_dataset
    except Exception as e:
        print(f'Error loading train: {e}')
        return None


def test_valid_split():
    """Test loading valid split."""
    print('\n=== Loading Valid Split ===')
    try:
        valid_dataset = AutismImageDataset('datasets/autism_images', split='valid')
        print(f'Valid dataset length: {len(valid_dataset)}')
        
        # Test first sample
        sample = valid_dataset[0]
        print(f'First sample label: {sample["label"]}')
        print(f'First sample class: {sample["class_name"]}')
        
        return valid_dataset
    except Exception as e:
        print(f'Error loading valid: {e}')
        return None


def test_test_split():
    """Test loading test split."""
    print('\n=== Loading Test Split ===')
    try:
        test_dataset = AutismImageDataset('datasets/autism_images', split='test')
        print(f'Test dataset length: {len(test_dataset)}')
        
        # Test first sample
        sample = test_dataset[0]
        print(f'First sample label: {sample["label"]}')
        print(f'First sample class: {sample["class_name"]}')
        
        return test_dataset
    except Exception as e:
        print(f'Error loading test: {e}')
        return None


def test_class_distribution(dataset, split_name):
    """Test class distribution in a dataset split."""
    if dataset is None:
        print(f'\n=== Skipping class distribution test for {split_name} (dataset not loaded) ===')
        return
    
    print(f'\n=== Class Distribution for {split_name} ===')
    class_counts = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        class_name = sample['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total = len(dataset)
    for class_name, count in class_counts.items():
        percentage = (count / total) * 100
        print(f'{class_name}: {count} samples ({percentage:.1f}%)')


def test_dataset_download():
    """Test dataset download functionality (without actually downloading)."""
    print('\n=== Testing Dataset Download Function ===')
    
    # Test check if dataset exists
    exists = AutismImageDataset._check_dataset_exists('datasets/autism_images')
    print(f'Dataset already exists: {exists}')
    
    if exists:
        print('Dataset is available for testing!')
    else:
        print('Dataset not found. You can download it using:')
        print('AutismImageDataset.download_dataset("datasets/autism_images")')


def main():
    """Run all tests."""
    print("Testing Autism Image Dataset Loader")
    print("=" * 50)
    
    # Test dataset verification
    dataset_exists = test_dataset_verification()
    
    if not dataset_exists:
        print("\nDataset not found. Please download it first using:")
        print("AutismImageDataset.download_dataset('datasets/autism_images')")
        return
    
    # Test dataset info
    info = test_dataset_info()
    
    # Calculate detailed statistics
    stats = calculate_detailed_statistics()
    
    # Test loading different splits
    train_dataset = test_train_split()
    valid_dataset = test_valid_split()
    test_dataset = test_test_split()
    
    # Test class distributions
    test_class_distribution(train_dataset, 'train')
    test_class_distribution(valid_dataset, 'valid')
    test_class_distribution(test_dataset, 'test')
    
    # Test download functionality
    test_dataset_download()
    
    print('\n=== Test Summary ===')
    print(f'Dataset exists: {dataset_exists}')
    print(f'Train split loaded: {train_dataset is not None}')
    print(f'Valid split loaded: {valid_dataset is not None}')
    print(f'Test split loaded: {test_dataset is not None}')
    
    if all([dataset_exists, train_dataset is not None, valid_dataset is not None, test_dataset is not None]):
        print('✅ All tests passed!')
    else:
        print('❌ Some tests failed!')


if __name__ == "__main__":
    main()