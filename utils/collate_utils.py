import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional


def unified_collate_fn(batch: List[Dict], 
                      pad_sequences: bool = False,
                      handle_numpy: bool = False,
                      handle_none: bool = False) -> Dict[str, Any]:
    """
    Unified collate function that can handle all dataloader requirements.
    
    Args:
        batch: List of sample dictionaries
        pad_sequences: Whether to pad variable-length sequences (for LMVD, CremaD, RAVDESS, PTSD)
        handle_numpy: Whether to handle numpy arrays (for WESAD)
        handle_none: Whether to handle None values (for WESAD)
    
    Returns:
        Collated batch dictionary
    """
    if not batch:
        return {}
    
    collated = {}
    
    # Get all unique keys from all samples
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())
    
    for key in all_keys:
        # Handle missing keys by using None or default values
        values = []
        for sample in batch:
            if key in sample:
                values.append(sample[key])
            else:
                values.append(None)
        
        # Handle None values
        if handle_none and any(v is None for v in values):
            collated[key] = values
            continue
            
        # Handle tensors
        if any(isinstance(v, torch.Tensor) for v in values if v is not None):
            # Filter out None values for tensor processing
            valid_values = [v for v in values if v is not None and isinstance(v, torch.Tensor)]
            if valid_values:
                if pad_sequences:
                    # Pad sequences to the same length (for variable-length data)
                    max_len = max(v.shape[0] for v in valid_values)
                    padded = [
                        F.pad(v, (0, 0, 0, max_len - v.shape[0])) if v.shape[0] < max_len else v
                        for v in valid_values
                    ]
                    collated[key] = torch.stack(padded)
                else:
                    # Simple stacking for fixed-length data
                    collated[key] = torch.stack(valid_values)
            else:
                collated[key] = values
                
        # Handle numpy arrays
        elif handle_numpy and any(isinstance(v, np.ndarray) for v in values if v is not None):
            try:
                # Convert numpy arrays to tensors and stack
                valid_values = [v for v in values if v is not None]
                tensor_values = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in valid_values]
                collated[key] = torch.stack(tensor_values)
            except:
                # Fall back to list if stacking fails
                collated[key] = values
                
        # Handle dictionaries (metadata)
        elif any(isinstance(v, dict) for v in values if v is not None):
            valid_values = [v for v in values if v is not None and isinstance(v, dict)]
            if valid_values:
                # Get all subkeys from all dictionaries
                all_subkeys = set()
                for v in valid_values:
                    all_subkeys.update(v.keys())
                
                collated[key] = {}
                for subkey in all_subkeys:
                    subvalues = []
                    for v in valid_values:
                        if subkey in v:
                            subvalues.append(v[subkey])
                        else:
                            subvalues.append(None)
                    collated[key][subkey] = subvalues
            else:
                collated[key] = values
            
        # Handle all other types (strings, ints, etc.)
        else:
            collated[key] = values
    
    return collated


# Convenience functions for specific dataloaders
def lmvd_collate_fn(batch):
    """Collate function for LMVD dataset with padding."""
    return unified_collate_fn(batch, pad_sequences=True)


def mosei_collate_fn(batch):
    """Collate function for MOSEI dataset."""
    return unified_collate_fn(batch, pad_sequences=False)


def wesad_collate_fn(batch):
    """Collate function for WESAD dataset with numpy and None handling."""
    return unified_collate_fn(batch, pad_sequences=False, handle_numpy=True, handle_none=True)


def esconv_collate_fn(batch):
    """Collate function for ESConv dataset."""
    return unified_collate_fn(batch, pad_sequences=False)


def cremad_collate_fn(batch):
    """Collate function for CremaD dataset with padding."""
    return unified_collate_fn(batch, pad_sequences=True)


def ravdess_collate_fn(batch):
    """Collate function for RAVDESS dataset with padding."""
    return unified_collate_fn(batch, pad_sequences=True)


def ptsditw_collate_fn(batch):
    """Collate function for PTSD in the Wild dataset with padding."""
    return unified_collate_fn(batch, pad_sequences=True)


def ch_simsv2_collate_fn(batch):
    """Collate function for CH_SIMSv2 dataset with padding."""
    return unified_collate_fn(batch, pad_sequences=True) 