#!/bin/bash
# Training script for Diabetic Retinopathy Detection

set -e

# Configuration
CONFIG_DIR="configs"
EXPERIMENT_NAME="dr_efficientnet_b3"
LOG_DIR="experiments/logs"
CHECKPOINT_DIR="experiments/checkpoints"

# Create directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --gpus)
            CUDA_VISIBLE_DEVICES="$2"
            export CUDA_VISIBLE_DEVICES
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Diabetic Retinopathy Detection - Training"
echo "============================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Config: ${CONFIG_FILE:-default}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "============================================"

# Run training
if [ -n "$CONFIG_FILE" ]; then
    python -m src.training.train --config "$CONFIG_FILE"
else
    python -m src.training.train
fi

echo "============================================"
echo "Training complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Logs saved to: $LOG_DIR"
echo "============================================"
