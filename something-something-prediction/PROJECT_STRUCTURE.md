# Something-Something V2 Frame Prediction Project Structure

## Overview
This project implements a text-conditioned video prediction model for human-object interaction frame prediction using the Something-Something V2 dataset. Given an observed frame from a human-object interaction and a natural-language description of the ongoing action, the model predicts the next frame of that sequence.

## Project Structure

```
something-something-prediction/
├── README.md                           # Project overview and usage instructions
├── requirements.txt                    # Python dependencies
├── train.py                           # Main training script
├── evaluate.py                        # Evaluation script with SSIM/PSNR metrics
├── dataset.py                         # Dataset class for Something-Something V2
├── utils.py                           # Utility functions
├── run_training.sh                    # Script to run training with default parameters
├── run_evaluation.sh                  # Script to run evaluation
├── PROJECT_STRUCTURE.md               # This document
├── models/                            # Model definitions (if needed)
│   └── (future use)
├── data/                              # Data processing scripts
│   └── prepare_dataset.py             # Script to prepare the dataset
└── output_model/                      # Default output directory for trained models (not in repo)
```

## Key Components

### 1. Training Script (`train.py`)
- Fine-tunes a Stable Diffusion model with InstructPix2Pix modifications
- Takes current frame and action description as input
- Predicts the frame 20 steps ahead (approximately 1.67 seconds at 12fps)
- Supports mixed precision training and gradient checkpointing
- Includes validation during training

### 2. Dataset Module (`dataset.py`)
- Handles loading of Something-Something V2 dataset from HuggingFace
- Filters for specific task categories: move_object, drop_object, cover_object
- Preprocesses video frames and action descriptions
- Provides data loaders for training and evaluation

### 3. Evaluation Script (`evaluate.py`)
- Evaluates trained models using SSIM and PSNR metrics
- Supports evaluation by task categories
- Saves results and sample predictions
- Provides detailed metrics for each task

### 4. Utility Functions (`utils.py`)
- Helper functions for image processing
- Model checkpointing and loading
- Configuration management
- Training utilities

### 5. Run Scripts
- `run_training.sh`: Simplified command-line interface for training
- `run_evaluation.sh`: Simplified command-line interface for evaluation

## Training Process

The model is trained using the following approach:
1. Load pre-trained InstructPix2Pix model
2. Fine-tune on Something-Something V2 dataset
3. Input: Current frame + Action description text
4. Output: Predicted frame 20 steps ahead (1.67 seconds)
5. Uses MSE loss between predicted and target frames

## Data Format

The model expects:
- Input: Current frame image (at time t)
- Condition: Action description text
- Target: Frame at time t+20 (1.67 seconds later at 12fps)

## Task Categories

The model focuses on three specific human-object interaction tasks:
- **move_object**: Horizontal translation of objects (e.g., "moving something from left to right")
- **drop_object**: Object transitions from support to free-fall (e.g., "dropping something onto something")
- **cover_object**: Manipulation to partially or fully cover another object (e.g., "covering something with something")

## Evaluation Metrics

The model is evaluated using:
- **SSIM** (Structural Similarity Index): Measures structural similarity between predicted and ground truth frames
- **PSNR** (Peak Signal-to-Noise Ratio): Measures the ratio between maximum possible power and noise power

## Requirements

- Python 3.8+
- PyTorch
- Diffusers library
- Transformers
- Datasets library
- OpenCV
- CUDA-compatible GPU (recommended for training)

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
bash run_training.sh --model timbrooks/instruct-pix2pix-00-22-42 --dataset /path/to/dataset --output ./my_model
```

### Evaluation
```bash
bash run_evaluation.sh --model ./my_model --dataset /path/to/dataset --num-samples 200
```

### Direct Usage
```bash
python train.py --pretrained_model_name_or_path timbrooks/instruct-pix2pix-00-22-42 --dataset_dir /path/to/dataset --output_dir ./output
python evaluate.py --model_path ./output --dataset_dir /path/to/dataset
```

## Expected Results

The model should learn to predict future frames based on:
1. Visual context from the current frame
2. Semantic understanding from the action description
3. Motion patterns specific to each task category

Performance is expected to vary by task:
- `move_object`: Good performance due to predictable motion patterns
- `drop_object`: Moderate performance due to physics-based motion
- `cover_object`: Good performance for static final states