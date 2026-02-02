#!/bin/bash
# Evaluation script for Diabetic Retinopathy Detection

set -e

# Default values
CHECKPOINT=""
TEST_CSV="data/processed/test.csv"
DATA_DIR="data/processed"
OUTPUT_DIR="experiments/results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --test-csv)
            TEST_CSV="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tta)
            USE_TTA=1
            shift
            ;;
        --gradcam)
            GENERATE_GRADCAM=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate checkpoint
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Usage: ./evaluate.sh --checkpoint path/to/checkpoint.ckpt"
    exit 1
fi

mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Diabetic Retinopathy Detection - Evaluation"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Test CSV: $TEST_CSV"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Run evaluation
python -c "
import sys
sys.path.insert(0, '.')

from src.training.validate import evaluate_checkpoint
from src.evaluation.metrics import print_classification_report
from src.utils.config_loader import load_config

# Evaluate
metrics = evaluate_checkpoint(
    checkpoint_path='$CHECKPOINT',
    config_path=None,
    test_csv='$TEST_CSV',
    data_dir='$DATA_DIR',
)

# Print results
print('\n' + '='*50)
print('EVALUATION RESULTS')
print('='*50)
print(f\"Accuracy: {metrics['accuracy']:.4f}\")
print(f\"Kappa: {metrics['kappa']:.4f}\")
print(f\"F1 (Macro): {metrics['f1_macro']:.4f}\")
print(f\"AUC (Macro): {metrics.get('auc_macro', 0):.4f}\")
print('='*50)
"

echo "============================================"
echo "Evaluation complete!"
echo "============================================"
