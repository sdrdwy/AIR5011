#!/bin/bash

# Script to run the Something-Something V2 frame prediction training
# Usage: ./run_training.sh [model_name] [dataset_dir] [output_dir]

set -e  # Exit on any error

# Default values
MODEL_NAME="timbrooks/instruct-pix2pix-00-22-42"
DATASET_DIR="./data"
OUTPUT_DIR="./output_model"
RESOLUTION=256
BATCH_SIZE=4
NUM_EPOCHS=50
LEARNING_RATE=5e-6
MAX_TRAIN_STEPS=10000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -s|--max-steps)
            MAX_TRAIN_STEPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -m, --model MODEL_NAME     Pretrained model name (default: timbrooks/instruct-pix2pix-00-22-42)"
            echo "  -d, --dataset DATASET_DIR  Dataset directory (default: ./data)"
            echo "  -o, --output OUTPUT_DIR    Output directory (default: ./output_model)"
            echo "  -r, --resolution RES       Image resolution (default: 256)"
            echo "  -b, --batch-size SIZE      Batch size (default: 4)"
            echo "  -e, --epochs EPOCHS        Number of epochs (default: 50)"
            echo "  -l, --learning-rate LR     Learning rate (default: 5e-6)"
            echo "  -s, --max-steps STEPS      Maximum training steps (default: 10000)"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Something-Something V2 frame prediction training..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "Resolution: $RESOLUTION"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_TRAIN_STEPS"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the training
accelerate launch --mixed_precision="fp16" train.py \
    --pretrained_model_name_or_path="$MODEL_NAME" \
    --dataset_dir="$DATASET_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --resolution="$RESOLUTION" \
    --train_batch_size="$BATCH_SIZE" \
    --num_train_epochs="$NUM_EPOCHS" \
    --max_train_steps="$MAX_TRAIN_STEPS" \
    --learning_rate="$LEARNING_RATE" \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --logging_dir="$OUTPUT_DIR/logs" \
    --report_to="tensorboard" \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=2

echo "Training completed! Model saved to $OUTPUT_DIR"