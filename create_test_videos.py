"""
Script to create test videos for Something-Something V2 dataset.
This creates simple animated videos to test our dataset loader.
"""

import cv2
import numpy as np
import os

def create_simple_video(video_path, duration=4, fps=12, resolution=256):
    """
    Create a simple animated video with a moving object.
    
    Args:
        video_path: Path to save the video
        duration: Duration of the video in seconds
        fps: Frames per second
        resolution: Resolution of the video (height and width)
    """
    total_frames = duration * fps
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM compatible
    out = cv2.VideoWriter(video_path, fourcc, fps, (resolution, resolution))
    
    for frame_idx in range(total_frames):
        # Create a blank white background
        frame = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
        
        # Calculate position of the moving circle
        # Move from left to right (for move_object task)
        x = int((frame_idx / total_frames) * (resolution - 50)) + 25
        y = resolution // 2
        
        # Draw a colored circle (representing an object)
        color = (0, 0, 255)  # Red circle
        cv2.circle(frame, (x, y), 20, color, -1)
        
        # Add a trail effect for movement visualization
        for i in range(1, 4):
            if frame_idx - i * 3 >= 0:
                prev_x = int(((frame_idx - i * 3) / total_frames) * (resolution - 50)) + 25
                alpha = 0.3 - i * 0.1  # Decreasing opacity for trail
                if alpha > 0:
                    overlay = frame.copy()
                    cv2.circle(overlay, (prev_x, y), 15, (0, 0, 255), -1)
                    frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)
        
        # Write the frame
        out.write(frame)
    
    # Release everything
    out.release()

def create_drop_video(video_path, duration=4, fps=12, resolution=256):
    """
    Create a video with an object dropping down (for drop_object task).
    """
    total_frames = duration * fps
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM compatible
    out = cv2.VideoWriter(video_path, fourcc, fps, (resolution, resolution))
    
    for frame_idx in range(total_frames):
        # Create a blank white background
        frame = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
        
        # Calculate position of the falling circle
        # Start at top and fall down with acceleration
        progress = frame_idx / total_frames
        x = resolution // 2
        y = int(progress * 0.8 * resolution) + 25  # Start from top, fall to near bottom
        
        # Draw a colored circle (representing an object)
        color = (0, 128, 0)  # Green circle
        cv2.circle(frame, (x, y), 20, color, -1)
        
        # Add a trail effect for drop visualization
        for i in range(1, 4):
            if frame_idx - i * 3 >= 0:
                prev_progress = (frame_idx - i * 3) / total_frames
                prev_y = int(prev_progress * 0.8 * resolution) + 25
                alpha = 0.3 - i * 0.1  # Decreasing opacity for trail
                if alpha > 0:
                    overlay = frame.copy()
                    cv2.circle(overlay, (x, prev_y), 15, (0, 128, 0), -1)
                    frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)
        
        # Write the frame
        out.write(frame)
    
    # Release everything
    out.release()

def create_cover_video(video_path, duration=4, fps=12, resolution=256):
    """
    Create a video with an object covering another (for cover_object task).
    """
    total_frames = duration * fps
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM compatible
    out = cv2.VideoWriter(video_path, fourcc, fps, (resolution, resolution))
    
    for frame_idx in range(total_frames):
        # Create a blank white background
        frame = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
        
        # Draw a stationary blue square (object to be covered)
        cv2.rectangle(frame, (resolution//2 - 40, resolution//2 + 20), 
                     (resolution//2 + 40, resolution//2 + 60), (255, 0, 0), -1)
        
        # Calculate position of the covering circle
        progress = frame_idx / total_frames
        x = resolution // 2
        y = int((1 - progress) * 0.6 * resolution) + 25  # Start from top, move down to cover
        
        # Draw a colored circle (covering object)
        color = (128, 0, 128)  # Purple circle
        cv2.circle(frame, (x, y), 30, color, -1)
        
        # Write the frame
        out.write(frame)
    
    # Release everything
    out.release()

def main():
    # Create a directory for test videos
    os.makedirs('/workspace/data/something-something-v2', exist_ok=True)
    
    # Create test videos for each category
    print("Creating test videos...")
    
    # Create videos for move_object category
    create_simple_video('/workspace/data/something-something-v2/000001.webm')
    create_simple_video('/workspace/data/something-something-v2/000002.webm')
    create_simple_video('/workspace/data/something-something-v2/000010.webm')
    create_simple_video('/workspace/data/something-something-v2/000045.webm')
    
    # Create videos for drop_object category
    create_drop_video('/workspace/data/something-something-v2/000005.webm')
    create_drop_video('/workspace/data/something-something-v2/000015.webm')
    create_drop_video('/workspace/data/something-something-v2/000030.webm')
    create_drop_video('/workspace/data/something-something-v2/000050.webm')
    
    # Create videos for cover_object category
    create_cover_video('/workspace/data/something-something-v2/000007.webm')
    create_cover_video('/workspace/data/something-something-v2/000020.webm')
    create_cover_video('/workspace/data/something-something-v2/000035.webm')
    create_cover_video('/workspace/data/something-something-v2/000055.webm')
    
    print("Test videos created successfully!")
    print("Videos created:")
    print("- 000001.webm, 000002.webm, 000010.webm, 000045.webm: Moving objects")
    print("- 000005.webm, 000015.webm, 000030.webm, 000050.webm: Dropping objects") 
    print("- 000007.webm, 000020.webm, 000035.webm, 000055.webm: Covering objects")

if __name__ == "__main__":
    main()