# Human-Object Interaction Frame Prediction

This project implements a text-conditioned video prediction model for human-object interaction frame prediction using the Something-Something V2 dataset. Given an observed frame from a human-object interaction and a natural-language description of the ongoing action, the model predicts the next frame of that sequence.

## Overview

The project implements an InstructPix2Pix-based approach to predict future frames in human-object interactions. It specifically focuses on three representative tasks:
- `move_object`: Horizontal translation of objects
- `drop_object`: Object transitions from support to free-fall
- `cover_object`: Manipulation of an object to partially or fully cover another

## Requirements

- Python 3.8+
- PyTorch
- Diffusers
- Transformers
- Datasets
- OpenCV

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The model can be configured by modifying the parameters in the training script:

- `--model_name`: Pretrained model name (e.g., "timbrooks/instruct-pix2pix-00-22-42")
- `--dataset_dir`: Path to the processed dataset directory
- `--output_dir`: Directory to save the trained model
- `--resolution`: Image resolution (minimum 96x96, default 256x256)
- `--batch_size`: Training batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for training

## Usage

### Training
```bash
python train.py --model_name "timbrooks/instruct-pix2pix-00-22-42" --dataset_dir "/path/to/dataset" --output_dir "./output_model"
```

### Evaluation
```bash
python evaluate.py --model_path "./output_model" --dataset_dir "/path/to/dataset"
```

## Dataset Processing

The model expects the Something-Something V2 dataset to be processed into the following format:
- Each video sample should be extracted into frames
- For each sample, we need:
  - Current frame (input)
  - Target frame (21 frames ahead)
  - Action description text
- The dataset should be organized by task categories (move_object, drop_object, cover_object)

## Model Architecture

The model is based on Stable Diffusion with InstructPix2Pix modifications:
- Takes current frame and action description as input
- Generates the predicted future frame
- Uses text conditioning to guide the frame prediction

## Evaluation Metrics

The model is evaluated using:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

## Project Structure

```
something-something-prediction/
├── README.md
├── requirements.txt
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── dataset.py            # Dataset processing
├── utils.py              # Utility functions
├── models/               # Model definitions
│   └── instruct_pix2pix.py
└── data/                 # Data processing scripts
    └── prepare_dataset.py
```

## Results

The model is trained to predict the 21st frame (1.75 seconds ahead at 12fps) given the current frame and action description. It achieves competitive results on the three main task categories.

## Acknowledgements

This project is developed for the MDS5122/AIR5011 Final Project at The Chinese University of Hong Kong, Shenzhen.