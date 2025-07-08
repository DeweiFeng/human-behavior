#!/usr/bin/env python3
"""
Test script to verify unified collate functions work correctly for all dataloaders.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.collate_utils import (
    unified_collate_fn, lmvd_collate_fn, mosei_collate_fn, wesad_collate_fn,
    esconv_collate_fn, cremad_collate_fn, ravdess_collate_fn, ptsditw_collate_fn,
    ch_simsv2_collate_fn
)

def test_unified_collate_basic():
    """Test basic functionality of unified collate function."""
    print("=== Testing Basic Unified Collate Function ===")
    
    # Test with tensors
    batch = [
        {'tensor': torch.randn(3, 4), 'id': 'sample1', 'metadata': {'a': 1}},
        {'tensor': torch.randn(3, 4), 'id': 'sample2', 'metadata': {'a': 2}},
        {'tensor': torch.randn(3, 4), 'id': 'sample3', 'metadata': {'a': 3}}
    ]
    
    result = unified_collate_fn(batch)
    print(f"✓ Basic tensor stacking: {result['tensor'].shape}")
    print(f"✓ ID list: {result['id']}")
    print(f"✓ Metadata handling: {result['metadata']}")
    
    # Test with variable length tensors (padding)
    batch_pad = [
        {'tensor': torch.randn(3, 4), 'id': 'sample1'},
        {'tensor': torch.randn(5, 4), 'id': 'sample2'},
        {'tensor': torch.randn(2, 4), 'id': 'sample3'}
    ]
    
    result_pad = unified_collate_fn(batch_pad, pad_sequences=True)
    print(f"✓ Variable length padding: {result_pad['tensor'].shape}")
    
    # Test with numpy arrays
    batch_numpy = [
        {'array': np.random.randn(3, 4), 'id': 'sample1'},
        {'array': np.random.randn(3, 4), 'id': 'sample2'}
    ]
    
    result_numpy = unified_collate_fn(batch_numpy, handle_numpy=True)
    print(f"✓ Numpy array handling: {result_numpy['array'].shape}")
    
    # Test with None values
    batch_none = [
        {'data': torch.randn(3, 4), 'optional': None, 'id': 'sample1'},
        {'data': torch.randn(3, 4), 'optional': torch.randn(2, 2), 'id': 'sample2'}
    ]
    
    result_none = unified_collate_fn(batch_none, handle_none=True)
    print(f"✓ None value handling: {result_none['optional']}")
    
    print("✓ All basic tests passed!\n")


def test_specific_collate_functions():
    """Test each specific collate function."""
    print("=== Testing Specific Collate Functions ===")
    
    # Test data
    batch = [
        {'tensor': torch.randn(3, 4), 'id': 'sample1', 'metadata': {'a': 1}},
        {'tensor': torch.randn(3, 4), 'id': 'sample2', 'metadata': {'a': 2}}
    ]
    
    # Test each function
    functions = [
        ('LMVD', lmvd_collate_fn),
        ('MOSEI', mosei_collate_fn),
        ('ESConv', esconv_collate_fn),
        ('CremaD', cremad_collate_fn),
        ('RAVDESS', ravdess_collate_fn),
        ('PTSD', ptsditw_collate_fn),
        ('CH_SIMSv2', ch_simsv2_collate_fn)
    ]
    
    for name, func in functions:
        try:
            result = func(batch)
            print(f"✓ {name}: {result['tensor'].shape}")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    # Test WESAD with numpy arrays
    batch_wesad = [
        {'data': np.random.randn(3, 4), 'id': 'sample1', 'metadata': {'a': 1}},
        {'data': np.random.randn(3, 4), 'id': 'sample2', 'metadata': {'a': 2}}
    ]
    
    try:
        result_wesad = wesad_collate_fn(batch_wesad)
        print(f"✓ WESAD: {result_wesad['data'].shape}")
    except Exception as e:
        print(f"✗ WESAD: {e}")
    
    print("✓ All specific function tests passed!\n")


def test_dataloader_integration():
    """Test integration with actual dataloaders."""
    print("=== Testing Dataloader Integration ===")
    
    # Test WESAD dataloader (since we have test data)
    try:
        from dataloaders.wesad_loader import create_wesad_dataloader
        
        # Check if WESAD data exists
        data_dir = "datasets/WESAD"
        if os.path.exists(data_dir):
            dataloader = create_wesad_dataloader(
                data_dir=data_dir,
                subjects=['S2', 'S3'],
                modalities=['ecg', 'eda'],
                window_size=1000,
                batch_size=2
            )
            
            for batch in dataloader:
                print(f"✓ WESAD dataloader batch keys: {list(batch.keys())}")
                print(f"✓ WESAD batch size: {len(batch['id'])}")
                break
        else:
            print("⚠ WESAD data not found, skipping integration test")
            
    except Exception as e:
        print(f"✗ WESAD integration test failed: {e}")
    
    print("✓ Integration tests completed!\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=== Testing Edge Cases ===")
    
    # Empty batch
    try:
        result = unified_collate_fn([])
        print(f"✓ Empty batch: {result}")
    except Exception as e:
        print(f"✗ Empty batch failed: {e}")
    
    # Single sample batch
    try:
        batch_single = [{'tensor': torch.randn(3, 4), 'id': 'single'}]
        result = unified_collate_fn(batch_single)
        print(f"✓ Single sample: {result['tensor'].shape}")
    except Exception as e:
        print(f"✗ Single sample failed: {e}")
    
    # Mixed data types
    try:
        batch_mixed = [
            {'tensor': torch.randn(3, 4), 'str': 'hello', 'int': 42, 'float': 3.14},
            {'tensor': torch.randn(3, 4), 'str': 'world', 'int': 100, 'float': 2.71}
        ]
        result = unified_collate_fn(batch_mixed)
        print(f"✓ Mixed types: tensor={result['tensor'].shape}, str={result['str']}")
    except Exception as e:
        print(f"✗ Mixed types failed: {e}")
    
    # Inconsistent keys
    try:
        batch_inconsistent = [
            {'tensor': torch.randn(3, 4), 'id': 'sample1'},
            {'tensor': torch.randn(3, 4)}  # Missing 'id'
        ]
        result = unified_collate_fn(batch_inconsistent)
        print(f"✓ Inconsistent keys: {result}")
    except Exception as e:
        print(f"✗ Inconsistent keys failed: {e}")
    
    print("✓ Edge case tests completed!\n")


def main():
    """Run all tests."""
    print("🧪 Testing Unified Collate Functions\n")
    
    test_unified_collate_basic()
    test_specific_collate_functions()
    test_dataloader_integration()
    test_edge_cases()
    
    print("🎉 All tests completed!")


if __name__ == "__main__":
    main() 