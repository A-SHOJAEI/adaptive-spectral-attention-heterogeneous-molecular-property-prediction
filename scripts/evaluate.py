#!/usr/bin/env python
"""Evaluation script for trained model."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.data.loader import (
    create_data_loaders,
    get_moleculenet_dataset,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.models.model import (
    AdaptiveSpectralAttentionModel,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.evaluation.metrics import (
    compute_metrics,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.evaluation.analysis import (
    analyze_per_task_performance,
    plot_confusion_matrix,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.utils.config import (
    load_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Tuple of (model, config).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Reconstruct model
    model = AdaptiveSpectralAttentionModel(
        num_node_features=config['model'].get('num_node_features', 9),
        num_tasks=config['model'].get('num_tasks', 1),
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        spectral_heads=config['model']['spectral_heads'],
        num_eigenvalues=config['model']['num_eigenvalues'],
        use_adaptive_filter=config['model']['use_adaptive_filter'],
        use_heterogeneous=config['model']['use_heterogeneous'],
        edge_types=config['model']['edge_types'],
        pool_type=config['model']['pool_type'],
        use_residual=config['model']['use_residual'],
        num_frequency_bands=config['spectral'].get('frequency_bands', 5)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model, config


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model.
        data_loader: Data loader for evaluation.
        device: Device to run evaluation on.

    Returns:
        Tuple of (predictions, labels).
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device)

            if not hasattr(batch, 'y') or batch.y is None:
                continue

            labels = batch.y.float()
            # Ensure labels are 2D [batch_size, num_tasks]
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)

            # Filter NaN labels
            valid_mask = ~torch.isnan(labels).any(dim=1)
            if not valid_mask.any():
                continue

            # Forward pass
            outputs = model(batch)
            outputs = outputs[valid_mask]
            labels = labels[valid_mask]

            # Convert to probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        logger.warning("No valid predictions generated")
        return None, None

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return all_preds, all_labels


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional, will use checkpoint config if not provided)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        model, config = load_model(args.checkpoint, device)

        # Override config if provided
        if args.config is not None:
            config = load_config(args.config)

        # Load dataset
        logger.info("Loading dataset...")
        dataset, num_tasks, num_node_features = get_moleculenet_dataset(
            config['data']['dataset_name'],
            config['data']['data_dir']
        )

        # Store in config for model loading
        config['model']['num_node_features'] = num_node_features
        config['model']['num_tasks'] = num_tasks

        # Create data loaders
        logger.info("Creating data loaders...")
        use_spectral = config['spectral'].get('use_spectral_attention', True)
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset,
            config,
            use_spectral=use_spectral
        )

        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        test_preds, test_labels = evaluate_model(model, test_loader, device)

        if test_preds is None:
            logger.error("Evaluation failed - no predictions generated")
            sys.exit(1)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_metrics(test_labels, test_preds)

        logger.info("\nTest Set Results:")
        logger.info("=" * 60)
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name:30s}: {metric_value:.4f}")
        logger.info("=" * 60)

        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\nMetrics saved to {metrics_path}")

        # Per-task analysis
        logger.info("\nAnalyzing per-task performance...")
        task_performance = analyze_per_task_performance(test_labels, test_preds)

        logger.info("\nPer-Task Performance:")
        logger.info(task_performance.to_string(index=False))

        # Save per-task results
        task_perf_path = output_dir / "per_task_performance.csv"
        task_performance.to_csv(task_perf_path, index=False)
        logger.info(f"Per-task performance saved to {task_perf_path}")

        # Plot confusion matrix
        logger.info("\nGenerating confusion matrix...")
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(test_labels, test_preds, save_path=str(cm_path))

        # Evaluate on validation set
        logger.info("\nEvaluating on validation set...")
        val_preds, val_labels = evaluate_model(model, val_loader, device)

        if val_preds is not None:
            val_metrics = compute_metrics(val_labels, val_preds)

            logger.info("\nValidation Set Results:")
            logger.info("=" * 60)
            for metric_name, metric_value in val_metrics.items():
                logger.info(f"{metric_name:30s}: {metric_value:.4f}")
            logger.info("=" * 60)

            # Save validation metrics
            val_metrics_path = output_dir / "validation_metrics.json"
            with open(val_metrics_path, 'w') as f:
                json.dump(val_metrics, f, indent=2)

        # Save predictions
        predictions_path = output_dir / "test_predictions.npz"
        np.savez(
            predictions_path,
            predictions=test_preds,
            labels=test_labels
        )
        logger.info(f"\nPredictions saved to {predictions_path}")

        logger.info("\nEvaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
