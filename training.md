# Training Guide

Detailed instructions for training diabetic retinopathy classification models.

## Prerequisites

1. **Hardware**: NVIDIA GPU with 8GB+ VRAM (tested on V100, A100, RTX 3090)
2. **Dataset**: Preprocessed fundus images with CSV annotations
3. **Environment**: Python 3.10+, CUDA 11.8+

## Data Preparation

### Expected Format

```
data/processed/
├── images/
│   ├── train/
│   │   ├── image_001.png
│   │   └── ...
│   ├── val/
│   └── test/
├── train.csv
├── val.csv
└── test.csv
```

### CSV Format

```csv
image_path,diagnosis
images/train/image_001.png,0
images/train/image_002.png,2
...
```

### Preprocessing Recommendations

1. **Crop to circular ROI**: Remove black borders
2. **Resize**: 512x512 or 768x768 depending on GPU memory
3. **Quality filter**: Remove blurry/low-quality images
4. **Balance check**: Log class distribution before training

## Training Workflow

### 1. Configure Experiment

Edit `configs/training.yaml` or create experiment-specific config:

```yaml
training:
  epochs: 50
  batch_size: 32  # Reduce if OOM

optimizer:
  lr: 1.0e-4  # Start conservative

loss:
  name: focal  # Better for imbalanced data
  focal_gamma: 2.0
```

### 2. Run Training

Basic:
```bash
python -m src.training.train
```

With custom config:
```bash
python -m src.training.train --config configs/experiment_v1.yaml
```

Multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.training.train
```

### 3. Monitor Progress

```bash
tensorboard --logdir experiments/logs
```

Key metrics to watch:
- `val_kappa`: Primary metric, should increase steadily
- `val_loss`: Should decrease, watch for overfitting
- `train_loss` vs `val_loss`: Gap indicates overfitting

### 4. Evaluate Best Checkpoint

```bash
./scripts/evaluate.sh --checkpoint experiments/checkpoints/best.ckpt
```

## Hyperparameter Recommendations

### Model Selection

| Model | Accuracy | Speed | Memory | Best For |
|-------|----------|-------|--------|----------|
| EfficientNet-B3 | High | Fast | 12GB | Default choice |
| EfficientNet-B5 | Higher | Medium | 24GB | Large datasets |
| ResNet50 | Medium | Fast | 8GB | Baselines |
| ViT-Base | High | Slow | 16GB | Research |

### Learning Rate Schedule

Recommended: Cosine annealing with warmup

```yaml
scheduler:
  name: cosine
  warmup_epochs: 5
  min_lr: 1.0e-7
```

### Data Augmentation

Default augmentation is aggressive. For smaller datasets, consider:
- Increase rotation limit (180°)
- Enable mixup (alpha=0.2)
- Use heavier coarse dropout

### Handling Class Imbalance

DR datasets are typically imbalanced (class 0 dominates). Options:

1. **Weighted sampling** (default, enabled)
2. **Focal loss** with gamma=2
3. **Class weights** in loss function
4. **Oversampling** minority classes

## Common Issues

### Out of Memory

- Reduce `batch_size`
- Use gradient accumulation: `accumulate_grad_batches: 2`
- Switch to smaller model variant

### Slow Convergence

- Check learning rate (try 3e-4)
- Ensure pretrained weights loaded
- Verify data normalization matches ImageNet

### Poor Validation Performance

- Check for data leakage between splits
- Increase regularization (dropout, weight decay)
- Add more augmentation
- Verify class balance in validation set

### NaN Loss

- Reduce learning rate
- Check for corrupted images
- Enable gradient clipping: `gradient_clip_val: 1.0`

## Experiment Tracking

### Local Setup

TensorBoard logs saved to `experiments/logs/`

### Weights & Biases Integration

1. Install: `pip install wandb`
2. Login: `wandb login`
3. Update training config to use WandbLogger

## Reproducibility Checklist

- [ ] Fixed random seed in config
- [ ] Deterministic mode enabled
- [ ] Same Python/PyTorch versions
- [ ] Full config saved with experiment
- [ ] Data split reproducible (seeded)

## Next Steps After Training

1. **Evaluate on held-out test set**
2. **Generate Grad-CAM visualizations** for clinical review
3. **Compute referable DR metrics** (sensitivity at 95% specificity)
4. **Export to ONNX** for deployment
5. **Document model card** with performance characteristics
