"""
Utility functions for Something-Something V2 frame prediction project.
"""

import torch
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime


def save_config(config, output_dir, filename="config.json"):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save the config
        filename: Name of the config file
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, filename)
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)  # Test if value is JSON serializable
            serializable_config[key] = value
        except TypeError:
            serializable_config[key] = str(value)  # Convert to string if not serializable
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the config file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def resize_image(image, size=(256, 256)):
    """
    Resize an image to the specified size.
    
    Args:
        image: PIL Image or numpy array
        size: Target size as (width, height)
    
    Returns:
        Resized PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    return image.resize(size)


def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor to a PIL Image.
    
    Args:
        tensor: PyTorch tensor of shape (C, H, W) or (H, W, C)
    
    Returns:
        PIL Image
    """
    # Ensure tensor is on CPU and detached from computation graph
    tensor = tensor.cpu().detach()
    
    # Normalize to [0, 1] if needed
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and then to PIL
    if len(tensor.shape) == 4:  # Batch dimension
        tensor = tensor[0]  # Take first item in batch
    
    # Move channel dimension to the end if needed (C, H, W) -> (H, W, C)
    if tensor.shape[0] in [1, 3]:  # Assume channel-first format
        tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy and then to PIL
    np_image = (tensor.numpy() * 255).astype(np.uint8)
    
    # Handle grayscale images
    if np_image.shape[-1] == 1:
        np_image = np_image.squeeze(-1)
        return Image.fromarray(np_image, mode='L')
    else:
        return Image.fromarray(np_image)


def pil_to_tensor(pil_image):
    """
    Convert a PIL Image to a PyTorch tensor.
    
    Args:
        pil_image: PIL Image
    
    Returns:
        PyTorch tensor of shape (C, H, W) with values in [-1, 1]
    """
    # Convert PIL image to numpy array
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(np_image)
    
    # Permute dimensions if needed to get (C, H, W) format
    if len(tensor.shape) == 3 and tensor.shape[-1] in [1, 3]:
        tensor = tensor.permute(2, 0, 1)
    
    # Scale from [0, 1] to [-1, 1]
    tensor = 2 * tensor - 1
    
    return tensor


def calculate_metrics(pred_image, target_image):
    """
    Calculate SSIM and PSNR metrics between predicted and target images.
    
    Args:
        pred_image: Predicted image (PIL or numpy)
        target_image: Target image (PIL or numpy)
    
    Returns:
        dict: Dictionary containing 'ssim' and 'psnr' values
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Convert to numpy arrays if needed
    if isinstance(pred_image, Image.Image):
        pred_np = np.array(pred_image)
    else:
        pred_np = pred_image
    
    if isinstance(target_image, Image.Image):
        target_np = np.array(target_image)
    else:
        target_np = target_image
    
    # Calculate SSIM
    ssim_val = ssim(target_np, pred_np, channel_axis=2, data_range=255.0)
    
    # Calculate PSNR
    psnr_val = psnr(target_np, pred_np, data_range=255.0)
    
    return {'ssim': ssim_val, 'psnr': psnr_val}


def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create a directory for an experiment with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional custom name for the experiment
    
    Returns:
        Path to the created experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def log_training_info(step, loss, lr, epoch=None, total_steps=None):
    """
    Log training information in a consistent format.
    
    Args:
        step: Current training step
        loss: Current loss value
        lr: Current learning rate
        epoch: Current epoch (optional)
        total_steps: Total training steps (optional)
    """
    info_str = f"Step: {step}"
    if epoch is not None:
        info_str += f", Epoch: {epoch}"
    if total_steps is not None:
        info_str += f"/{total_steps}"
    info_str += f", Loss: {loss:.6f}, LR: {lr:.2e}"
    
    print(info_str)


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Get the appropriate device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: The device to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def save_model_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir, filename=None):
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        step: Current training step
        loss: Current loss
        checkpoint_dir: Directory to save the checkpoint
        filename: Custom filename (optional)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pth"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_model_checkpoint(model, optimizer, checkpoint_path):
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        tuple: (epoch, step, loss) from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss}")
    
    return epoch, step, loss


def format_time(seconds):
    """
    Format time in seconds to a human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


# Example usage and testing
if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test image conversion
    test_image = Image.new('RGB', (256, 256), color='red')
    tensor = pil_to_tensor(test_image)
    print(f"Original image size: {test_image.size}")
    print(f"Converted tensor shape: {tensor.shape}")
    
    # Test parameter counting
    import torch.nn as nn
    simple_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    params = count_parameters(simple_model)
    print(f"Number of parameters in simple model: {params}")
    
    print("Utility functions test completed.")