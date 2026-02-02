# Experiments Directory

This directory stores training artifacts, checkpoints, and results.

## Structure

```
experiments/
├── checkpoints/     # Model checkpoints
├── logs/           # TensorBoard logs and training logs
├── results/        # Evaluation results and visualizations
└── README.md
```

## Checkpoints

Model checkpoints are saved with the naming convention:
```
epoch={epoch:02d}-val_kappa={val_kappa:.4f}.ckpt
```

The best model is determined by validation quadratic weighted kappa.

## Viewing Training Logs

Launch TensorBoard to visualize training:

```bash
tensorboard --logdir experiments/logs
```

## Experiment Tracking

For production experiments, consider integrating:
- Weights & Biases (wandb)
- MLflow
- Neptune.ai

Configuration for W&B is available in the training config.
