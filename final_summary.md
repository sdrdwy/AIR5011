# MDS5122/AIR5011 Final Project - Human-Object Interaction Frame Prediction

## Project Overview

This repository contains the implementation of a text-conditioned video prediction model for the MDS5122/AIR5011 Final Project at The Chinese University of Hong Kong, Shenzhen. The project implements a model that predicts future frames in human-object interactions using the Something-Something V2 dataset.

## Problem Statement

Given several observed frames of a human-object interaction and a natural-language description of the ongoing action, the model must predict the next frame of that sequence. This task mimics the reasoning challenge of anticipating motion and visual change from linguistic and visual context—an ability relevant to robotics, AR/VR, and embodied AI.

## Approach

The solution uses a fine-tuned InstructPix2Pix model based on Stable Diffusion with the following key components:

1. **Model Architecture**: Modified UNet with 8 input channels (4 for current frame + 4 for latent representation)
2. **Training Data**: Something-Something V2 dataset with video frames and action descriptions
3. **Task Focus**: Three representative human-object interaction tasks:
   - `move_object`: Horizontal translation of objects
   - `drop_object`: Object transitions from support to free-fall
   - `cover_object`: Manipulation to partially or fully cover another object
4. **Prediction Horizon**: 21st frame prediction (1.75 seconds ahead at 12fps)

## Key Features

- **Text-Conditioned Prediction**: Uses natural language action descriptions to guide frame prediction
- **Multi-Task Learning**: Trained on three distinct interaction categories
- **Evaluation Metrics**: SSIM and PSNR for quantitative assessment
- **Scalable Architecture**: Supports various resolutions (minimum 96x96)

## Repository Structure

```
something-something-prediction/
├── README.md                           # Project overview
├── requirements.txt                    # Dependencies
├── train.py                           # Training script
├── evaluate.py                        # Evaluation script
├── dataset.py                         # Dataset handling
├── utils.py                           # Utility functions
├── run_training.sh                    # Training runner
├── run_evaluation.sh                  # Evaluation runner
├── PROJECT_STRUCTURE.md               # Detailed structure
└── data/                              # Data processing
    └── prepare_dataset.py
```

## Implementation Details

### Model Architecture
- Base: `timbrooks/instruct-pix2pix-00-22-42` (pre-trained InstructPix2Pix)
- Modified UNet with additional image conditioning channels
- Text encoder for action description processing
- Diffusion-based frame generation

### Training Configuration
- Image resolution: 256x256 (minimum 96x96 requirement met)
- Batch size: 4 with gradient accumulation
- Learning rate: 5e-6
- Mixed precision training (fp16) for efficiency
- Gradient checkpointing to manage memory usage

### Dataset Processing
- Loads Something-Something V2 from HuggingFace
- Filters for specific task categories
- Creates input-target pairs (current frame → frame + 20)
- Handles video-to-frame conversion

## Results and Evaluation

The model is evaluated using:
- **SSIM (Structural Similarity Index)**: Measures structural similarity between predicted and ground truth frames
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between maximum possible power and noise power

Evaluation is performed separately for each task category to assess performance across different types of interactions.

## Usage Instructions

### Setup
```bash
cd something-something-prediction
pip install -r requirements.txt
```

### Training
```bash
bash run_training.sh --model timbrooks/instruct-pix2pix-00-22-42 --dataset /path/to/dataset --output ./output_model
```

### Evaluation
```bash
bash run_evaluation.sh --model ./output_model --dataset /path/to/dataset --num-samples 100
```

## Technical Contributions

1. **Task-Specific Dataset Filtering**: Automatically identifies and categorizes videos based on action keywords
2. **Frame Prediction Pipeline**: Efficiently processes video sequences for temporal prediction
3. **Multi-Task Evaluation**: Assesses performance across different interaction types
4. **Memory-Efficient Training**: Uses gradient checkpointing and mixed precision for large model training

## Computational Requirements

- GPU with at least 12GB memory (for full training)
- CUDA-compatible hardware for acceleration
- Sufficient storage for model checkpoints and dataset
- Python 3.8+ environment

## Expected Outcomes

The model should demonstrate:
- Understanding of action semantics from text descriptions
- Ability to predict motion patterns specific to each task category
- Reasonable visual quality in predicted frames
- Quantitative performance metrics (SSIM/PSNR) for comparison across tasks

## Acknowledgments

This project was developed for the MDS5122/AIR5011 Final Project at The Chinese University of Hong Kong, Shenzhen. The implementation builds upon the Hugging Face Diffusers library and the Something-Something V2 dataset.

## References

- Something-Something V2 Dataset: https://developer.qualcomm.com/software/ai-datasets/something-something
- InstructPix2Pix: https://huggingface.co/timbrooks/instruct-pix2pix-00-22-42
- Diffusers Library: https://github.com/huggingface/diffusers