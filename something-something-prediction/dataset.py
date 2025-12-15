"""
Dataset class for Something-Something V2 frame prediction task.
This class handles the loading and preprocessing of the Something-Something V2 dataset
for the frame prediction task.
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
import cv2


class SomethingSomethingDataset(Dataset):
    """
    Dataset class for Something-Something V2 frame prediction.
    Expects the dataset to be organized with video frames and action descriptions.
    """
    def __init__(self, dataset_dir, resolution=256, max_samples=None, frame_step=20):
        """
        Args:
            dataset_dir: Path to the dataset directory
            resolution: Resolution to resize images to
            max_samples: Maximum number of samples to load (for debugging)
            frame_step: Number of frames to skip to get the target frame (default 20, which is ~1.67s at 12fps)
        """
        self.dataset_dir = dataset_dir
        self.resolution = resolution
        self.max_samples = max_samples
        self.frame_step = frame_step
        self.data = []
        
        # Define the action categories we want to focus on
        self.move_object_keywords = [
            "moving something", "pushing something", "pulling something"
        ]
        self.drop_object_keywords = [
            "dropping something", "letting something fall", "lifting something up completely, then letting it drop down"
        ]
        self.cover_object_keywords = [
            "covering something", "putting something on top of something"
        ]
        
        # Load the dataset
        self._load_dataset()
        
        if max_samples:
            self.data = self.data[:max_samples]
    
    def _load_dataset(self):
        """
        Load the Something-Something V2 dataset from HuggingFace or local directory.
        """
        print("Loading Something-Something V2 dataset...")
        
        # Try to load from HuggingFace first
        try:
            dataset = load_dataset("HuggingFaceM4/something_something_v2", split="train")
            print(f"Loaded {len(dataset)} samples from HuggingFace dataset")
            
            # Filter for the specific tasks we want
            for idx, sample in enumerate(dataset):
                text = sample['text'].lower()
                
                # Check if the action belongs to one of our target categories
                if (any(keyword in text for keyword in self.move_object_keywords) or
                    any(keyword in text for keyword in self.drop_object_keywords) or
                    any(keyword in text for keyword in self.cover_object_keywords)):
                    
                    # For now, we'll store the video_id and text description
                    # In a real implementation, you would extract frames from the video
                    self.data.append({
                        'video_id': sample['video_id'],
                        'text': sample['text'],
                        'label': sample['label']
                    })
                    
                    if self.max_samples and len(self.data) >= self.max_samples:
                        break
        except Exception as e:
            print(f"Could not load from HuggingFace: {e}")
            print("Please ensure you have access to the dataset or provide a local path")
            # For now, we'll create some dummy data for demonstration
            for i in range(100):  # Create 100 dummy samples
                self.data.append({
                    'video_id': f"dummy_{i}",
                    'text': f"dummy action description {i}",
                    'label': i % 3  # Simulate 3 classes
                })
    
    def _extract_frames(self, video_path, target_frame_idx):
        """
        Extract frames from a video file.
        This is a placeholder implementation - in practice you would use a video processing library.
        """
        # In a real implementation, you would extract frames from the video
        # For now, we'll return dummy frames
        frames = []
        for i in range(target_frame_idx + 1):  # Generate frames up to target
            # Create a dummy frame (in practice, load from video)
            frame = np.random.randint(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8)
            frames.append(Image.fromarray(frame))
        return frames
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Returns:
            dict: Contains 'input_image', 'target_image', and 'edit_prompt'
        """
        sample = self.data[idx]
        
        # In a real implementation, you would:
        # 1. Load the video
        # 2. Extract frames
        # 3. Select current frame (input) and target frame (current + frame_step)
        
        # For demonstration, create dummy frames
        input_image = np.random.randint(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8)
        target_image = np.random.randint(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8)
        
        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)
        
        return {
            'input_image': input_image,
            'target_image': target_image,
            'edit_prompt': sample['text']  # Use the action description as the edit prompt
        }
    
    def set_transform(self, transform):
        """
        Set the transform function to be applied to samples.
        """
        self.transform = transform


def get_task_specific_dataset(dataset_dir, task_name, resolution=256, max_samples=None):
    """
    Get a dataset filtered for a specific task.
    
    Args:
        dataset_dir: Path to the dataset directory
        task_name: One of 'move_object', 'drop_object', 'cover_object'
        resolution: Resolution to resize images to
        max_samples: Maximum number of samples to load
    """
    if task_name not in ['move_object', 'drop_object', 'cover_object']:
        raise ValueError(f"task_name must be one of ['move_object', 'drop_object', 'cover_object'], got {task_name}")
    
    # Create a dataset that only includes samples for the specified task
    full_dataset = SomethingSomethingDataset(dataset_dir, resolution, max_samples)
    
    # Filter based on task-specific keywords
    task_keywords = {
        'move_object': [
            "moving something", "pushing something", "pulling something"
        ],
        'drop_object': [
            "dropping something", "letting something fall", "lifting something up completely, then letting it drop down"
        ],
        'cover_object': [
            "covering something", "putting something on top of something"
        ]
    }
    
    keywords = task_keywords[task_name]
    filtered_data = []
    
    for sample in full_dataset.data:
        text = sample['text'].lower()
        if any(keyword in text for keyword in keywords):
            filtered_data.append(sample)
    
    # Create a new dataset with filtered data
    dataset = SomethingSomethingDataset(dataset_dir, resolution, max_samples)
    dataset.data = filtered_data
    
    return dataset


def get_combined_task_dataset(dataset_dir, task_names, resolution=256, max_samples_per_task=None):
    """
    Get a combined dataset for multiple tasks.
    
    Args:
        dataset_dir: Path to the dataset directory
        task_names: List of task names to include
        resolution: Resolution to resize images to
        max_samples_per_task: Maximum number of samples per task
    """
    combined_data = []
    
    for task_name in task_names:
        task_dataset = get_task_specific_dataset(dataset_dir, task_name, resolution, max_samples_per_task)
        combined_data.extend(task_dataset.data)
    
    # Create a new dataset with combined data
    dataset = SomethingSomethingDataset(dataset_dir, resolution, max_samples_per_task)
    dataset.data = combined_data
    
    return dataset


# Example usage:
if __name__ == "__main__":
    # Example of how to use the dataset
    dataset = SomethingSomethingDataset(
        dataset_dir="/path/to/something-something-v2", 
        resolution=256,
        max_samples=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Input image shape: {sample['input_image'].size}")
    print(f"Target image shape: {sample['target_image'].size}")
    print(f"Edit prompt: {sample['edit_prompt']}")
    
    # Example of getting task-specific datasets
    move_dataset = get_task_specific_dataset(
        "/path/to/something-something-v2", 
        "move_object", 
        resolution=256,
        max_samples=50
    )
    print(f"Move object dataset size: {len(move_dataset)}")