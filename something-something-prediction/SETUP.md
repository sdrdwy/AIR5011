# Setup Guide for Something-Something V2 Frame Prediction

This guide explains how to set up the environment and run the frame prediction model on the Something-Something V2 dataset.

## Dataset Structure

The code expects the Something-Something V2 dataset to be organized as follows:

```
/path/to/something-something-v2/
├── 1.webm
├── 2.webm
├── ...
├── 100000.webm
├── something-something-v2-labels.json
├── something-something-v2-train.json
└── [other annotation files]
```

## Environment Setup

1. **Install Python Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

2. **Install Additional Dependencies** (if needed):
   ```bash
   pip install diffusers[torch] accelerate transformers datasets opencv-python scikit-image huggingface_hub xformers wandb ftfy tqdm
   ```

## Data Preparation

1. **Download the Dataset**: 
   - Download the Something-Something V2 dataset from the official source
   - You'll need the video files and annotation files

2. **Prepare the Dataset**:
   ```bash
   python data/prepare_dataset.py \
       --dataset_dir /path/to/something-something-v2 \
       --output_dir /path/to/processed_dataset \
       --tasks move_object drop_object cover_object \
       --max_samples 1000
   ```

## Training

To train the model:

```bash
python train.py \
    --pretrained_model_name_or_path "timbrooks/instruct-pix2pix-00-22-42" \
    --dataset_dir /path/to/processed_dataset \
    --output_dir ./output_model \
    --resolution 256 \
    --train_batch_size 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6 \
    --max_train_samples 100 \
    --gradient_accumulation_steps 4
```

## Project-Specific Configuration

The code is configured to work with the three specific action classes from the project:

1. **move_object**: Actions like "moving something from left to right", "pushing something from right to left"
2. **drop_object**: Actions like "dropping something onto something", "letting something fall down" 
3. **cover_object**: Actions like "covering something with something", "putting something on top of something"

The dataset loader specifically filters for these action types as specified in the project requirements.

## Expected Behavior

- The model will extract frames from each video
- Use the first frame as input and a frame 20 steps ahead (at 12fps ≈ 1.67 seconds) as the target
- Use the action description as the text prompt for conditional generation
- Train an InstructPix2Pix-based model to predict the future frame based on current frame and action description

## Evaluation

After training, you can evaluate the model using SSIM and PSNR metrics as required by the project:

```bash
python evaluate.py --model_path ./output_model --dataset_dir /path/to/processed_dataset
```

## Notes

- The frame extraction and processing might be slow for large datasets
- Make sure you have enough disk space for extracted frames if using the frame extraction approach
- For the project, you need to generate at least 100 observations per task (300 total)
- The model should achieve reasonable SSIM and PSNR scores for successful frame prediction