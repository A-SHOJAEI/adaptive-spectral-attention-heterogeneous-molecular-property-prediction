"""Evaluation metrics for molecular property prediction."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro"
) -> float:
    """
    Compute ROC-AUC score.

    Args:
        y_true: Ground truth labels of shape [n_samples, n_tasks].
        y_pred: Predicted probabilities of shape [n_samples, n_tasks].
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        ROC-AUC score.
    """
    try:
        # Handle single task
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_true = y_true.ravel()
            y_pred = y_pred.ravel()

            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                logger.warning("Only one class present in y_true. ROC-AUC not defined.")
                return 0.5

            return roc_auc_score(y_true, y_pred)

        # Multi-task
        scores = []
        for task_idx in range(y_true.shape[1]):
            y_true_task = y_true[:, task_idx]
            y_pred_task = y_pred[:, task_idx]

            # Filter out NaN values
            valid_mask = ~np.isnan(y_true_task)
            if not valid_mask.any():
                continue

            y_true_task = y_true_task[valid_mask]
            y_pred_task = y_pred_task[valid_mask]

            # Check if we have both classes
            if len(np.unique(y_true_task)) < 2:
                continue

            score = roc_auc_score(y_true_task, y_pred_task)
            scores.append(score)

        if not scores:
            return 0.5

        return float(np.mean(scores))

    except Exception as e:
        logger.warning(f"Error computing ROC-AUC: {e}")
        return 0.5


def compute_average_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute average precision score.

    Args:
        y_true: Ground truth labels of shape [n_samples, n_tasks].
        y_pred: Predicted probabilities of shape [n_samples, n_tasks].

    Returns:
        Average precision score.
    """
    try:
        # Handle single task
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            y_true = y_true.ravel()
            y_pred = y_pred.ravel()

            if len(np.unique(y_true)) < 2:
                return 0.0

            return average_precision_score(y_true, y_pred)

        # Multi-task
        scores = []
        for task_idx in range(y_true.shape[1]):
            y_true_task = y_true[:, task_idx]
            y_pred_task = y_pred[:, task_idx]

            # Filter out NaN values
            valid_mask = ~np.isnan(y_true_task)
            if not valid_mask.any():
                continue

            y_true_task = y_true_task[valid_mask]
            y_pred_task = y_pred_task[valid_mask]

            if len(np.unique(y_true_task)) < 2:
                continue

            score = average_precision_score(y_true_task, y_pred_task)
            scores.append(score)

        if not scores:
            return 0.0

        return float(np.mean(scores))

    except Exception as e:
        logger.warning(f"Error computing average precision: {e}")
        return 0.0


def compute_spectral_alignment_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spectral_complexities: Optional[np.ndarray] = None
) -> float:
    """
    Compute spectral alignment score.

    This custom metric measures how well the model's predictions align
    with the spectral complexity of molecules. Better models should
    perform consistently across different complexity levels.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.
        spectral_complexities: Optional array of spectral complexity scores.

    Returns:
        Spectral alignment score between 0 and 1.
    """
    if spectral_complexities is None:
        # If no complexity scores provided, fall back to standard accuracy
        y_pred_binary = (y_pred > 0.5).astype(int)
        return accuracy_score(y_true.ravel(), y_pred_binary.ravel())

    try:
        # Divide samples into complexity bins
        num_bins = 5
        complexity_bins = np.linspace(0, 1, num_bins + 1)

        bin_accuracies = []

        for i in range(num_bins):
            low = complexity_bins[i]
            high = complexity_bins[i + 1]

            # Get samples in this complexity bin
            mask = (spectral_complexities >= low) & (spectral_complexities < high)

            if not mask.any():
                continue

            y_true_bin = y_true[mask]
            y_pred_bin = y_pred[mask]

            # Compute accuracy for this bin
            y_pred_binary = (y_pred_bin > 0.5).astype(int)
            acc = accuracy_score(y_true_bin.ravel(), y_pred_binary.ravel())
            bin_accuracies.append(acc)

        if not bin_accuracies:
            return 0.0

        # Alignment score is based on:
        # 1. Average accuracy across bins
        # 2. Inverse of standard deviation (consistency across complexities)
        avg_acc = np.mean(bin_accuracies)
        std_acc = np.std(bin_accuracies)

        # Penalize high variance (inconsistent performance)
        alignment_score = avg_acc * (1.0 - min(std_acc, 0.5))

        return float(np.clip(alignment_score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"Error computing spectral alignment score: {e}")
        return 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spectral_complexities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.
        spectral_complexities: Optional spectral complexity scores.

    Returns:
        Dictionary of metric names and values.
    """
    # Convert to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Flatten for single-task or average multi-task
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    y_pred_binary_flat = y_pred_binary.ravel()

    # Remove NaN values
    valid_mask = ~np.isnan(y_true_flat)
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]
    y_pred_binary_flat = y_pred_binary_flat[valid_mask]

    metrics = {}

    # ROC-AUC
    metrics['roc_auc'] = compute_roc_auc(y_true, y_pred)

    # Average Precision
    metrics['avg_precision'] = compute_average_precision(y_true, y_pred)

    # Spectral Alignment Score
    metrics['spectral_alignment_score'] = compute_spectral_alignment_score(
        y_true,
        y_pred,
        spectral_complexities
    )

    # Accuracy
    try:
        metrics['accuracy'] = accuracy_score(y_true_flat, y_pred_binary_flat)
    except:
        metrics['accuracy'] = 0.0

    # F1 Score
    try:
        metrics['f1'] = f1_score(y_true_flat, y_pred_binary_flat, average='macro', zero_division=0)
    except:
        metrics['f1'] = 0.0

    # Precision
    try:
        metrics['precision'] = precision_score(y_true_flat, y_pred_binary_flat, average='macro', zero_division=0)
    except:
        metrics['precision'] = 0.0

    # Recall
    try:
        metrics['recall'] = recall_score(y_true_flat, y_pred_binary_flat, average='macro', zero_division=0)
    except:
        metrics['recall'] = 0.0

    return metrics
