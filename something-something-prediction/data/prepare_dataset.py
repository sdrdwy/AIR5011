"""
Data preparation script for Something-Something V2 dataset.
This script downloads and preprocesses the dataset for frame prediction task.
"""
import os
import json
import argparse
from datasets import load_dataset
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import tempfile
import subprocess


def extract_frames_from_video(video_path, output_dir, frame_step=20):
    """
    Extract frames from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_step: Extract every nth frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ffmpeg to extract frames
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps=1/{frame_step}',  # Extract every frame_step seconds
        f'{output_dir}/frame_%06d.jpg',
        '-y'  # Overwrite output files without asking
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Extracted frames from {video_path}")
    except subprocess.CalledProcessError:
        print(f"Error extracting frames from {video_path}")
        return []
    
    # Return list of extracted frame paths
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    return [os.path.join(output_dir, f) for f in frame_files]


def prepare_dataset(dataset_dir, output_dir, task_filter=None, max_samples=None):
    """
    Prepare the Something-Something V2 dataset for frame prediction.
    
    Args:
        dataset_dir: Path to the raw dataset directory
        output_dir: Path to save the processed dataset
        task_filter: List of task names to include (e.g., ['move_object', 'drop_object'])
        max_samples: Maximum number of samples to process
    """
    print(f"Preparing dataset from {dataset_dir} to {output_dir}")
    
    # Define task keywords
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
    
    # If no task filter is specified, use all tasks
    if task_filter is None:
        task_filter = list(task_keywords.keys())
    
    # Validate task filter
    for task in task_filter:
        if task not in task_keywords:
            raise ValueError(f"Unknown task: {task}. Valid tasks: {list(task_keywords.keys())}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Load the dataset
    print("Loading Something-Something V2 dataset...")
    try:
        dataset = load_dataset("HuggingFaceM4/something_something_v2", split="train")
        print(f"Loaded {len(dataset)} samples from HuggingFace dataset")
    except Exception as e:
        print(f"Could not load dataset from HuggingFace: {e}")
        print("Please make sure you have the proper access to the dataset")
        return
    
    # Prepare filtered data
    processed_samples = []
    sample_count = 0
    
    for idx, sample in enumerate(dataset):
        if max_samples and sample_count >= max_samples:
            break
            
        text = sample['text'].lower()
        
        # Check if this sample belongs to one of our target tasks
        is_target_task = False
        for task in task_filter:
            if any(keyword in text for keyword in task_keywords[task]):
                is_target_task = True
                break
        
        if not is_target_task:
            continue
        
        # For this implementation, we'll simulate the process
        # In a real scenario, you would extract frames from the video
        print(f"Processing sample {idx}: {sample['text']}")
        
        # Add to processed samples
        processed_samples.append({
            'video_id': sample['video_id'],
            'text': sample['text'],
            'label': sample['label'],
            'task': next(task for task in task_filter if any(keyword in text for keyword in task_keywords[task]))
        })
        
        sample_count += 1
        
        if sample_count % 100 == 0:
            print(f"Processed {sample_count} samples...")
    
    # Save the processed samples metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(processed_samples, f, indent=2)
    
    print(f"Dataset preparation completed. Processed {len(processed_samples)} samples.")
    print(f"Metadata saved to {metadata_path}")
    
    # Create a sample data structure
    # In practice, you would create actual frame pairs
    sample_data = {
        'task_filter': task_filter,
        'total_samples': len(processed_samples),
        'tasks': list(task_filter),
        'description': 'Processed Something-Something V2 dataset for frame prediction',
        'format': {
            'input_image': 'Current frame at time t',
            'target_image': 'Frame at time t+20 (1.67 seconds later at 12fps)',
            'edit_prompt': 'Action description text'
        }
    }
    
    config_path = os.path.join(output_dir, 'dataset_config.json')
    with open(config_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Dataset configuration saved to {config_path}")


def create_sample_pairs_from_video(video_path, output_dir, sample_id, frame_step=20):
    """
    Create input-target pairs from a video for training.
    This is a helper function to process individual videos.
    """
    # Extract frames from video
    temp_dir = os.path.join(output_dir, f'temp_{sample_id}')
    frame_paths = extract_frames_from_video(video_path, temp_dir, frame_step=1)  # Extract all frames first
    
    if len(frame_paths) < frame_step + 1:
        print(f"Video {video_path} is too short for frame_step={frame_step}")
        return None
    
    # Create input-target pairs
    pairs = []
    for i in range(len(frame_paths) - frame_step):
        input_frame_path = frame_paths[i]
        target_frame_path = frame_paths[i + frame_step]
        
        pair = {
            'input_frame': input_frame_path,
            'target_frame': target_frame_path,
            'frame_step': frame_step
        }
        pairs.append(pair)
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare Something-Something V2 dataset for frame prediction.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the raw dataset directory (not used in this implementation since we load from HuggingFace)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the processed dataset"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        choices=['move_object', 'drop_object', 'cover_object'],
        default=['move_object', 'drop_object', 'cover_object'],
        help="List of tasks to include in the dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process (default: 1000)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        task_filter=args.tasks,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()