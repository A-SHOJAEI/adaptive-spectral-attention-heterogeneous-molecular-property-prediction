"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: List[Dict[str, float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        val_metrics: List of validation metric dictionaries per epoch.
        save_path: Optional path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_losses, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Metric curves
    if val_metrics:
        metric_names = list(val_metrics[0].keys())
        for metric_name in metric_names:
            values = [m.get(metric_name, 0.0) for m in val_metrics]
            axes[1].plot(epochs, values, label=metric_name, marker='o')

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.
        save_path: Optional path to save the plot.
    """
    # Convert to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Flatten
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred_binary.ravel()

    # Remove NaN
    valid_mask = ~np.isnan(y_true_flat)
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]

    # Compute confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_per_task_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze performance per task for multi-task prediction.

    Args:
        y_true: Ground truth labels of shape [n_samples, n_tasks].
        y_pred: Predicted probabilities of shape [n_samples, n_tasks].
        task_names: Optional list of task names.

    Returns:
        DataFrame with per-task metrics.
    """
    from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.evaluation.metrics import (
        compute_roc_auc,
        compute_average_precision,
    )

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    num_tasks = y_true.shape[1]

    if task_names is None:
        task_names = [f"Task_{i}" for i in range(num_tasks)]

    results = []

    for task_idx in range(num_tasks):
        y_true_task = y_true[:, task_idx]
        y_pred_task = y_pred[:, task_idx]

        # Filter NaN
        valid_mask = ~np.isnan(y_true_task)
        if not valid_mask.any():
            continue

        y_true_task = y_true_task[valid_mask]
        y_pred_task = y_pred_task[valid_mask]

        # Compute metrics
        try:
            roc_auc = compute_roc_auc(
                y_true_task.reshape(-1, 1),
                y_pred_task.reshape(-1, 1)
            )
        except:
            roc_auc = 0.0

        try:
            avg_prec = compute_average_precision(
                y_true_task.reshape(-1, 1),
                y_pred_task.reshape(-1, 1)
            )
        except:
            avg_prec = 0.0

        # Binary predictions
        y_pred_binary = (y_pred_task > 0.5).astype(int)

        try:
            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(y_true_task, y_pred_binary)
            f1 = f1_score(y_true_task, y_pred_binary, zero_division=0)
        except:
            accuracy = 0.0
            f1 = 0.0

        results.append({
            'Task': task_names[task_idx],
            'ROC-AUC': roc_auc,
            'Avg Precision': avg_prec,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Num Samples': len(y_true_task)
        })

    df = pd.DataFrame(results)
    return df


def save_results_summary(
    metrics: Dict[str, float],
    config: Dict,
    save_path: str
) -> None:
    """
    Save results summary to file.

    Args:
        metrics: Dictionary of metrics.
        config: Configuration dictionary.
        save_path: Path to save the summary.
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Metrics:\n")
        f.write("-" * 40 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name:30s}: {value:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Configuration:\n")
        f.write("=" * 80 + "\n")

        # Write key config sections
        for section in ['data', 'model', 'training']:
            if section in config:
                f.write(f"\n{section.upper()}:\n")
                f.write("-" * 40 + "\n")
                for key, value in config[section].items():
                    f.write(f"  {key:28s}: {value}\n")

    logger.info(f"Results summary saved to {save_path}")
