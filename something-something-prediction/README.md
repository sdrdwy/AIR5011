# Human-Object Interaction Frame Prediction for Something-Something V2

This project implements a text-conditioned video prediction model for human-object interaction frame prediction using the Something-Something V2 dataset. Given an observed frame from a human-object interaction and a natural-language description of the ongoing action, the model predicts the next frame of that sequence.

## Overview

The project implements an InstructPix2Pix-based approach to predict future frames in human-object interactions. It specifically focuses on three representative tasks:

- `move_object`: Horizontal translation of objects (e.g., "Moving something from left to right", "Pushing something from right to left")
- `drop_object`: Object transitions from support to free-fall (e.g., "Dropping something onto something", "Letting something fall down") 
- `cover_object`: Manipulation of an object to partially or fully cover another (e.g., "Covering something with something", "Putting something on top of something")

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
python train.py --pretrained_model_name_or_path "timbrooks/instruct-pix2pix-00-22-42" --dataset_dir "/path/to/dataset" --output_dir "./output_model"
```

### Evaluation
```bash
python evaluate.py --model_path "./output_model" --dataset_dir "/path/to/dataset"
```

## Dataset Processing

The model expects the Something-Something V2 dataset to be processed into the following format:
- Video files in .webm format (1.webm, 2.webm, etc.) stored in the dataset directory
- Label file: `something-something-v2-labels.json` containing action category mappings
- Annotation file: `something-something-v2-train.json` containing video IDs and action templates
- The dataset should be organized by the three specific task categories mentioned above

The code will:
1. Load videos and their corresponding text descriptions from the annotation files
2. Filter for the three specific action classes required by the project:
   - move_object: "Moving something from left to right", "Pushing something from right to left", etc.
   - drop_object: "Dropping something onto something", "Letting something fall down", etc.
   - cover_object: "Covering something with something", "Putting something on top of something", etc.
3. Extract frames from each video using OpenCV
4. Use the first frame as input and the frame 20 steps ahead (≈1.67 seconds at 12fps) as the target
5. Use the action description as the text conditioning prompt

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
├── SETUP.md
├── requirements.txt
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── dataset.py            # Dataset processing
├── utils.py              # Utility functions
└── data/                 # Data processing scripts
    └── prepare_dataset.py
```

## Results

The model is trained to predict the 21st frame (1.75 seconds ahead at 12fps) given the current frame and action description. It achieves competitive results on the three main task categories.

## Project-Specific Implementation Details

This implementation is specifically designed for the MDS5122/AIR5011 Final Project requirements:

1. **Dataset**: Something-Something V2 with ~220K short human-object interaction videos
2. **Model**: Fine-tuned InstructPix2Pix model that accepts current frame and textual action instruction
3. **Tasks**: Focuses on three representative human-object interaction tasks:
   - move_object: Horizontal translation
   - drop_object: Vertical motion under gravity
   - cover_object: Static contact completion
4. **Resolution**: At least 96×96 (configurable up to higher resolutions)
5. **Frame Prediction**: Uses 20 observed frames as input and predicts the 21st frame as output
6. **Training**: Minimum 100 observations per task (300 total for all tasks)
7. **Evaluation**: Uses SSIM and PSNR metrics

## Acknowledgements

This project is developed for the MDS5122/AIR5011 Final Project at The Chinese University of Hong Kong, Shenzhen.