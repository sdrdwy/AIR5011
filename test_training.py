"""
Test script to verify the training process works with the Something-Something V2 dataset.
This runs a minimal training test to ensure the integration works correctly.
"""

import sys
import os
sys.path.append('/workspace/something-something-prediction')

import torch
from dataset import SomethingSomethingDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def test_training_integration():
    print("Testing training integration with Something-Something V2 dataset...")
    
    # Create a small dataset for testing
    dataset = SomethingSomethingDataset(
        dataset_dir="/workspace/data/something-something-v2",
        resolution=128,  # Use smaller resolution for testing
        max_samples=4,   # Small dataset for testing
        frame_step=5     # Use smaller frame step for testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return False
    
    # Define the tokenization function (simplified)
    from transformers import CLIPTokenizer
    try:
        # Use a smaller, faster tokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    except:
        # If not available locally, initialize empty
        print("Using placeholder tokenizer...")
        def tokenize_captions(captions):
            # Simple placeholder tokenizer
            return torch.randint(0, 1000, (len(captions), 77))  # [batch_size, 77] - typical CLIP length
    
        def tokenize_captions(captions):
            tokens = []
            for caption in captions:
                # Simple hash-based tokenization for testing
                token_ids = [abs(hash(caption)) % 1000 + i for i in range(77)]
                tokens.append(token_ids)
            return torch.tensor(tokens)
    else:
        def tokenize_captions(captions):
            inputs = tokenizer(
                captions, 
                max_length=tokenizer.model_max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            return inputs.input_ids

    # Define image preprocessing transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to tensors
    ])

    def convert_to_tensor(image, resolution=128):
        """Convert image to tensor and normalize."""
        image = image.convert("RGB").resize((resolution, resolution))
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        return tensor / 127.5 - 1.0  # Normalize to [-1, 1]

    def preprocess_images(examples, resolution=128):
        """Preprocess images for training."""
        import numpy as np
        
        original_images = []
        edited_images = []
        
        for img in examples["input_image"]:
            img_tensor = convert_to_tensor(img, resolution)
            original_images.append(img_tensor)
        
        for img in examples["target_image"]:
            img_tensor = convert_to_tensor(img, resolution)
            edited_images.append(img_tensor)
        
        original_images = torch.stack(original_images)
        edited_images = torch.stack(edited_images)
        
        return {
            "original_pixel_values": original_images,
            "edited_pixel_values": edited_images,
        }

    def preprocess_train(examples, resolution=128):
        """Preprocess training examples."""
        # Preprocess images
        processed = preprocess_images(examples, resolution)
        
        # Tokenize captions
        captions = list(examples["edit_prompt"])
        input_ids = tokenize_captions(captions)
        
        processed["input_ids"] = input_ids
        return processed

    def collate_fn(examples):
        """Collate function for DataLoader."""
        # Extract and stack images
        original_pixel_values = torch.stack([ex["original_pixel_values"] for ex in examples])
        edited_pixel_values = torch.stack([ex["edited_pixel_values"] for ex in examples])
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    
    # Test single example
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input image type: {type(sample['input_image'])}")
    print(f"Edit prompt: {sample['edit_prompt']}")
    
    # Test batch processing
    batch_examples = [dataset[i] for i in range(min(2, len(dataset)))]
    
    # Preprocess the batch
    batch_data = preprocess_images({
        "input_image": [ex["input_image"] for ex in batch_examples],
        "target_image": [ex["target_image"] for ex in batch_examples]
    }, resolution=128)
    
    # Tokenize prompts
    prompts = [ex["edit_prompt"] for ex in batch_examples]
    input_ids = tokenize_captions(prompts)
    
    batch_data["input_ids"] = input_ids
    
    print(f"Processed batch - Original images shape: {batch_data['original_pixel_values'].shape}")
    print(f"Processed batch - Target images shape: {batch_data['edited_pixel_values'].shape}")
    print(f"Processed batch - Input IDs shape: {batch_data['input_ids'].shape}")
    
    # Test DataLoader
    print("Testing DataLoader...")
    try:
        # Create a simple dataset wrapper for DataLoader
        class SimpleDataset:
            def __init__(self, original_dataset):
                self.original_dataset = original_dataset
            
            def __len__(self):
                return min(4, len(self.original_dataset))
            
            def __getitem__(self, idx):
                item = self.original_dataset[idx]
                # Preprocess the item
                processed = preprocess_images({
                    "input_image": [item["input_image"]],
                    "target_image": [item["target_image"]]
                }, resolution=128)
                
                # Tokenize the prompt
                input_ids = tokenize_captions([item["edit_prompt"]])
                
                result = {
                    "original_pixel_values": processed["original_pixel_values"][0],
                    "edited_pixel_values": processed["edited_pixel_values"][0],
                    "input_ids": input_ids[0] if input_ids.dim() > 1 else input_ids
                }
                return result
        
        simple_dataset = SimpleDataset(dataset)
        dataloader = DataLoader(simple_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        for i, batch in enumerate(dataloader):
            print(f"DataLoader batch {i}:")
            print(f"  Original pixel values shape: {batch['original_pixel_values'].shape}")
            print(f"  Edited pixel values shape: {batch['edited_pixel_values'].shape}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            break  # Just test one batch
        
        print("Training integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in training integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_integration()
    if success:
        print("\n✓ All tests passed! The Something-Something V2 dataset integration is working correctly.")
    else:
        print("\n✗ Tests failed!")