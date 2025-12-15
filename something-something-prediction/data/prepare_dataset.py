"""
Data preparation script for Something-Something V2 dataset.
This script processes the dataset for frame prediction task.
"""
import os
import json
import argparse
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
import subprocess
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, frame_step=20):
    """
    Extract frames from a video file using OpenCV.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_step: Extract every nth frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # Save frame as image
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        success = cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        if success:
            frames.append(frame_path)
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) > 0:
        print(f"Extracted {len(frames)} frames from {video_path}")
    else:
        print(f"Failed to extract frames from {video_path}")
    
    return frames


def create_sample_pairs_from_video(video_path, output_dir, sample_id, frame_step=20):
    """
    Create input-target pairs from a video for training.
    This is a helper function to process individual videos.
    """
    # Extract frames from video
    temp_dir = os.path.join(output_dir, f'temp_{sample_id}')
    frame_paths = extract_frames_from_video(video_path, temp_dir, frame_step=1)  # Extract all frames first
    
    if len(frame_paths) < frame_step + 1:
        print(f"Video {video_path} is too short for frame_step={frame_step} (only {len(frame_paths)} frames)")
        # Clean up temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
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


def prepare_dataset(dataset_dir, output_dir, task_filter=None, max_samples=None):
    """
    Prepare the Something-Something V2 dataset for frame prediction.
    
    Args:
        dataset_dir: Path to the raw dataset directory with video files
        output_dir: Path to save the processed dataset
        task_filter: List of task names to include (e.g., ['move_object', 'drop_object'])
        max_samples: Maximum number of samples to process
    """
    print(f"Preparing dataset from {dataset_dir} to {output_dir}")
    
    # Define task keywords based on the project requirements
    task_keywords = {
        'move_object': [
            "moving something from left to right", "pushing something from right to left",
            "moving something", "pushing something", "pulling something"
        ],
        'drop_object': [
            "dropping something onto something", "letting something fall down",
            "dropping something", "letting something fall"
        ],
        'cover_object': [
            "covering something with something", "putting something on top of something",
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
    
    # Look for label files
    label_file = os.path.join(dataset_dir, 'something-something-v2-labels.json')
    annotations_file = os.path.join(dataset_dir, 'something-something-v2-train.json')
    
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
        processed_samples = []
        sample_count = 0
        
        for annotation in tqdm(annotations, desc="Processing annotations"):
            if max_samples and sample_count >= max_samples:
                break
                
            video_id = str(annotation['id']).zfill(6)  # Format as 6-digit number with leading zeros
            text = annotation['template'].replace('[', '').replace(']', '')  # Clean template
            label = annotation['label_id']
            
            # Check if the action belongs to one of our target tasks
            text_lower = text.lower()
            is_target_task = False
            for task in task_filter:
                if any(keyword in text_lower for keyword in task_keywords[task]):
                    is_target_task = True
                    break
            
            if not is_target_task:
                continue
            
            # Check if the video file exists
            video_path = os.path.join(dataset_dir, f"{video_id}.webm")
            if os.path.exists(video_path):
                processed_samples.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'text': text,
                    'label': label,
                    'task': next(task for task in task_filter if any(keyword in text_lower for keyword in task_keywords[task]))
                })
                sample_count += 1
                
                if sample_count % 100 == 0:
                    print(f"Processed {sample_count} samples...")
    
    else:
        print(f"Label files not found at {label_file} or {annotations_file}")
        print("Looking for video files in the dataset directory...")
        
        # If label files don't exist, scan for video files
        video_extensions = ['.webm', '.mp4', '.avi', '.mov']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(dataset_dir).glob(f"*{ext}"))
        
        # Sort and take first few files if max_samples is set
        video_files = sorted(video_files)[:max_samples] if max_samples else sorted(video_files)
        
        print(f"Found {len(video_files)} video files")
        
        # For each video file, we'll create a basic dataset structure
        processed_samples = []
        for video_path in tqdm(video_files, desc="Processing videos"):
            video_id = video_path.stem  # filename without extension
            
            # Create a placeholder entry
            processed_samples.append({
                'video_id': video_id,
                'video_path': str(video_path),
                'text': f"action in video {video_id}",  # Placeholder text
                'label': 0,  # Placeholder label
                'task': 'unknown'  # Placeholder task
            })
    
    # Save the processed samples metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(processed_samples, f, indent=2)
    
    print(f"Dataset preparation completed. Processed {len(processed_samples)} samples.")
    print(f"Metadata saved to {metadata_path}")
    
    # Create a sample data structure
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


def main():
    parser = argparse.ArgumentParser(description="Prepare Something-Something V2 dataset for frame prediction.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the raw dataset directory containing video files"
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