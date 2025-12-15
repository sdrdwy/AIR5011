#!/usr/bin/env python
"""
Test script to verify the Something-Something V2 dataset implementation
"""
import os
import sys
sys.path.append('/workspace/something-something-prediction')

from dataset import SomethingSomethingDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def test_dataset():
    print("Testing Something-Something V2 dataset implementation...")
    
    # Test with a sample dataset directory (this will use dummy data if actual dataset not found)
    dataset_dir = "/workspace/something-something-v2"  # This directory may not exist, so dummy data will be used
    
    try:
        dataset = SomethingSomethingDataset(
            dataset_dir=dataset_dir,
            resolution=256,
            max_samples=10  # Test with 10 samples
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Print first few samples to verify
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}:")
            print(f"  Input image shape: {sample['input_image'].size}")
            print(f"  Target image shape: {sample['target_image'].size}")
            print(f"  Edit prompt: {sample['edit_prompt']}")
            print()
        
        # Test with DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        print("Batch test:")
        print(f"  Input batch shape: {batch['input_image'][0].size if batch['input_image'] else 'N/A'}")
        print(f"  Target batch shape: {batch['target_image'][0].size if batch['target_image'] else 'N/A'}")
        print(f"  Edit prompts: {batch['edit_prompt'][:2]}")
        
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()