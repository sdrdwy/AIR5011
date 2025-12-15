#!/bin/bash

# Script to run the Something-Something V2 frame prediction evaluation
# Usage: ./run_evaluation.sh [model_path] [dataset_dir]

set -e  # Exit on any error

# Default values
MODEL_PATH="./output_model"
DATASET_DIR="./data"
NUM_SAMPLES=100
OUTPUT_DIR="./evaluation_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_DIR="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --by-task)
            BY_TASK=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -m, --model MODEL_PATH     Path to trained model (default: ./output_model)"
            echo "  -d, --dataset DATASET_DIR  Dataset directory (default: ./data)"
            echo "  -n, --num-samples N        Number of samples to evaluate (default: 100)"
            echo "  -o, --output OUTPUT_DIR    Output directory (default: ./evaluation_results)"
            echo "      --by-task              Evaluate separately for each task category"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Something-Something V2 frame prediction evaluation..."
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the evaluation
if [ "$BY_TASK" = true ]; then
    echo "Evaluating by task categories..."
    python evaluate.py \
        --model_path="$MODEL_PATH" \
        --dataset_dir="$DATASET_DIR" \
        --num_samples="$NUM_SAMPLES" \
        --output_dir="$OUTPUT_DIR" \
        --by_task
else
    echo "Evaluating all samples..."
    python evaluate.py \
        --model_path="$MODEL_PATH" \
        --dataset_dir="$DATASET_DIR" \
        --num_samples="$NUM_SAMPLES" \
        --output_dir="$OUTPUT_DIR"
fi

echo "Evaluation completed! Results saved to $OUTPUT_DIR"