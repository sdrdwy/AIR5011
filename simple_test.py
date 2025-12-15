"""
Simple test to verify the dataset loading works with the Something-Something V2 dataset.
"""

import sys
sys.path.append('/workspace/something-something-prediction')

from dataset import SomethingSomethingDataset

def test_dataset():
    print("Testing Something-Something dataset loader...")
    
    # Create dataset instance
    dataset = SomethingSomethingDataset(
        dataset_dir="/workspace/data/something-something-v2",
        resolution=256,
        max_samples=5,  # Limit for testing
        frame_step=5  # Use smaller frame step for testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return
    
    print(f"Sample data keys: {dataset[0].keys()}")
    
    # Test getting a sample
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Input image type: {type(sample['input_image'])}")
        print(f"  Input image size: {sample['input_image'].size}")
        print(f"  Target image type: {type(sample['target_image'])}")
        print(f"  Target image size: {sample['target_image'].size}")
        print(f"  Edit prompt: {sample['edit_prompt']}")
    
    # Test task-specific datasets
    print("\nTesting task-specific datasets...")
    
    # Test move_object dataset
    move_dataset = SomethingSomethingDataset(
        dataset_dir="/workspace/data/something-something-v2",
        resolution=256,
        max_samples=10,
        frame_step=5
    )
    
    # Filter for move_object tasks
    move_data = []
    move_keywords = ["moving something", "pushing something", "pulling something"]
    for item in move_dataset.data:
        text = item['text'].lower()
        if any(keyword in text for keyword in move_keywords):
            move_data.append(item)
    
    print(f"Move object samples: {len(move_data)}")
    
    # Test drop_object dataset
    drop_dataset = SomethingSomethingDataset(
        dataset_dir="/workspace/data/something-something-v2",
        resolution=256,
        max_samples=10,
        frame_step=5
    )
    
    # Filter for drop_object tasks
    drop_data = []
    drop_keywords = ["dropping something", "letting something fall"]
    for item in drop_dataset.data:
        text = item['text'].lower()
        if any(keyword in text for keyword in drop_keywords):
            drop_data.append(item)
    
    print(f"Drop object samples: {len(drop_data)}")
    
    # Test cover_object dataset
    cover_dataset = SomethingSomethingDataset(
        dataset_dir="/workspace/data/something-something-v2",
        resolution=256,
        max_samples=10,
        frame_step=5
    )
    
    # Filter for cover_object tasks
    cover_data = []
    cover_keywords = ["covering something", "putting something on top of something"]
    for item in cover_dataset.data:
        text = item['text'].lower()
        if any(keyword in text for keyword in cover_keywords):
            cover_data.append(item)
    
    print(f"Cover object samples: {len(cover_data)}")
    
    print("\nDataset test completed successfully!")

if __name__ == "__main__":
    test_dataset()