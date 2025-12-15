"""
Test script to verify the Something-Something dataset loader works correctly
with the Something-Something V2 dataset.
"""

import sys
sys.path.append('/workspace/something-something-prediction')

from dataset import SomethingSomethingDataset
import torch
from torch.utils.data import DataLoader
from PIL import Image

def test_dataset():
    print("Testing Something-Something dataset loader...")
    
    # Create dataset instance
    dataset = SomethingSomethingDataset(
        dataset_dir="/workspace/data/something-something-v2",
        resolution=256,
        max_samples=20,  # Limit for testing
        frame_step=20  # Predict frame 20 steps ahead
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return
    
    print(f"Sample data keys: {dataset[0].keys()}")
    
    # Test getting a sample
    sample = dataset[0]
    print(f"Input image type: {type(sample['input_image'])}")
    print(f"Input image size: {sample['input_image'].size}")
    print(f"Target image type: {type(sample['target_image'])}")
    print(f"Target image size: {sample['target_image'].size}")
    print(f"Edit prompt: {sample['edit_prompt']}")
    
    # Test with DataLoader using the same collate function as training
    def tokenize_captions(captions):
        # Simple placeholder - in real training this would use a proper tokenizer
        return [len(caption) for caption in captions]  # Just return length as placeholder
    
    def preprocess_images(examples, resolution=256):
        import numpy as np
        import torch
        from torchvision import transforms
        
        # Convert PIL images to numpy arrays
        original_images = []
        edited_images = []
        
        for img in examples['input_image']:
            img_array = np.array(img.convert('RGB'))
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
            img_tensor = torch.from_numpy(img_array).float() / 255.0 * 2 - 1  # Normalize to [-1, 1]
            original_images.append(img_tensor)
        
        for img in examples['target_image']:
            img_array = np.array(img.convert('RGB'))
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
            img_tensor = torch.from_numpy(img_array).float() / 255.0 * 2 - 1  # Normalize to [-1, 1]
            edited_images.append(img_tensor)
        
        original_images = torch.stack(original_images)
        edited_images = torch.stack(edited_images)
        
        return {
            "original_pixel_values": original_images,
            "edited_pixel_values": edited_images,
        }
    
    def collate_fn(examples):
        # Preprocess the batch
        input_images = [ex['input_image'] for ex in examples]
        target_images = [ex['target_image'] for ex in examples]
        prompts = [ex['edit_prompt'] for ex in examples]
        
        # Process images
        batch_data = preprocess_images({
            'input_image': input_images,
            'target_image': target_images
        })
        
        # Tokenize prompts
        input_ids = tokenize_captions(prompts)
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 1, 1]
        
        batch_data["input_ids"] = input_ids_tensor
        return batch_data
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Original pixel values shape: {batch['original_pixel_values'].shape}")
        print(f"  Edited pixel values shape: {batch['edited_pixel_values'].shape}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        
        if i >= 2:  # Just test a few batches
            break
    
    print("Dataset test completed successfully!")

if __name__ == "__main__":
    test_dataset()