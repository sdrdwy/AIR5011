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
import subprocess
from pathlib import Path


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
        Load the Something-Something V2 dataset from local directory.
        """
        print("Loading Something-Something V2 dataset from local directory...")
        
        # Define the exact keywords for our three tasks as specified in the project description
        self.move_object_keywords = [
            "moving something from left to right", "pushing something from right to left",
            "moving something", "pushing something", "pulling something"
        ]
        self.drop_object_keywords = [
            "dropping something onto something", "letting something fall down",
            "dropping something", "letting something fall"
        ]
        self.cover_object_keywords = [
            "covering something with something", "putting something on top of something",
            "covering something", "putting something on top of something"
        ]
        
        # Look for label files in the dataset directory
        label_file = os.path.join(self.dataset_dir, 'something-something-v2-labels.json')
        annotations_file = os.path.join(self.dataset_dir, 'something-something-v2-train.json')
        
        if os.path.exists(label_file) and os.path.exists(annotations_file):
            print(f"Found label files: {label_file}, {annotations_file}")
            
            # Load the labels and annotations
            with open(label_file, 'r') as f:
                labels_map = json.load(f)
                
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
                
            # Create reverse mapping for labels
            label_to_text = {int(k): v for k, v in labels_map.items()}
            
            # Process annotations
            for annotation in annotations:
                video_id = str(annotation['id']).zfill(6)  # Format as 6-digit number with leading zeros
                text = annotation['template'].replace('[', '').replace(']', '')  # Clean template
                label = annotation['label_id']
                
                # Check if the action belongs to one of our target categories
                text_lower = text.lower()
                if (any(keyword in text_lower for keyword in self.move_object_keywords) or
                    any(keyword in text_lower for keyword in self.drop_object_keywords) or
                    any(keyword in text_lower for keyword in self.cover_object_keywords)):
                    
                    # Check if the video file exists
                    video_path = os.path.join(self.dataset_dir, f"{video_id}.webm")
                    if os.path.exists(video_path):
                        self.data.append({
                            'video_id': video_id,
                            'video_path': video_path,
                            'text': text,
                            'label': label
                        })
                        print(f"Added video: {video_id} with action: {text}")
                        
                        if self.max_samples and len(self.data) >= self.max_samples:
                            break
            print(f"Loaded {len(self.data)} valid samples from local dataset")
        else:
            print(f"Label files not found at {label_file} or {annotations_file}")
            print("Looking for video files in the dataset directory...")
            
            # If label files don't exist, scan for video files and try to match based on filenames
            video_extensions = ['.webm', '.mp4', '.avi', '.mov']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(Path(self.dataset_dir).glob(f"*{ext}"))
            
            # Sort and take first few files if max_samples is set
            video_files = sorted(video_files)[:self.max_samples] if self.max_samples else sorted(video_files)
            
            print(f"Found {len(video_files)} video files")
            
            # For each video file, we'll try to determine if it fits our criteria
            # In the absence of labels, we'll create a basic dataset structure
            for video_path in video_files:
                video_id = video_path.stem  # filename without extension
                
                # Since we don't have text descriptions, we'll use the file name as a placeholder
                # In practice, you'd want to load the actual labels from the annotations
                self.data.append({
                    'video_id': video_id,
                    'video_path': str(video_path),
                    'text': f"action in video {video_id}",  # Placeholder text
                    'label': 0  # Placeholder label
                })
                
                if self.max_samples and len(self.data) >= self.max_samples:
                    break
                    
            if len(self.data) == 0:
                print("No video files found, creating dummy data for demonstration")
                # Create some dummy data if no videos are found
                for i in range(min(100, self.max_samples or 100)):  # Create up to 100 dummy samples
                    self.data.append({
                        'video_id': f"dummy_{i:06d}",
                        'video_path': f"/dummy/path/dummy_{i:06d}.webm",
                        'text': f"dummy action description {i}",
                        'label': i % 3  # Simulate 3 classes
                    })
    
    def _extract_frames(self, video_path, target_frame_idx=None):
        """
        Extract frames from a video file using OpenCV.
        """
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            return []
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return []
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to the required resolution
            frame_resized = cv2.resize(frame_rgb, (self.resolution, self.resolution))
            
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_resized)
            frames.append(pil_frame)
            
            frame_idx += 1
            
            # If target_frame_idx is specified, we can stop early
            if target_frame_idx is not None and frame_idx > target_frame_idx:
                break
        
        cap.release()
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
        
        # Extract frames from the video
        frames = self._extract_frames(sample['video_path'])
        
        if len(frames) == 0:
            # If we can't extract frames, return dummy data
            input_image = np.random.randint(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8)
            target_image = np.random.randint(0, 255, (self.resolution, self.resolution, 3), dtype=np.uint8)
            input_image = Image.fromarray(input_image)
            target_image = Image.fromarray(target_image)
        elif len(frames) < self.frame_step + 1:
            # If video is too short, duplicate the last frame
            input_image = frames[0]
            target_image = frames[-1]
        else:
            # Select input frame (first frame) and target frame (frame_step frames ahead)
            input_image = frames[0]  # First frame as input
            target_frame_idx = min(self.frame_step, len(frames) - 1)  # Ensure we don't go out of bounds
            target_image = frames[target_frame_idx]
        
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