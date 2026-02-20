#!/usr/bin/env python
"""Training script for adaptive spectral attention model."""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

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
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.training.trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.utils.config import (
    load_config,
    save_config,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.evaluation.analysis import (
    plot_training_curves,
    save_results_summary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_mlflow(config: dict) -> None:
    """
    Setup MLflow tracking if enabled.

    Args:
        config: Configuration dictionary.
    """
    if config['logging'].get('use_mlflow', False):
        try:
            import mlflow

            experiment_name = config['logging'].get(
                'experiment_name',
                'adaptive_spectral_molecular'
            )
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()

            # Log parameters
            mlflow.log_params({
                'learning_rate': config['training']['learning_rate'],
                'batch_size': config['data']['batch_size'],
                'hidden_dim': config['model']['hidden_dim'],
                'num_layers': config['model']['num_layers'],
                'use_adaptive_filter': config['model']['use_adaptive_filter'],
                'use_heterogeneous': config['model']['use_heterogeneous'],
                'use_curriculum': config['curriculum']['use_curriculum'],
            })

            logger.info("MLflow tracking enabled")
            return True

        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")
            return False
    else:
        logger.info("MLflow tracking disabled")
        return False


def main() -> None:
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train adaptive spectral attention model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config['reproducibility'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    results_dir = Path(config['paths']['results_dir'])
    log_dir = Path(config['paths'].get('log_dir', 'logs'))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    use_mlflow = setup_mlflow(config)

    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset, num_tasks, num_node_features = get_moleculenet_dataset(
            config['data']['dataset_name'],
            config['data']['data_dir']
        )

        logger.info(f"Dataset: {len(dataset)} molecules, {num_tasks} tasks")

        # Store dataset properties in config for model loading later
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

        logger.info(
            f"Data loaders created - Train: {len(train_loader.dataset)}, "
            f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}"
        )

        # Create model
        logger.info("Creating model...")
        model = AdaptiveSpectralAttentionModel(
            num_node_features=num_node_features,
            num_tasks=num_tasks,
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

        model = model.to(device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {num_params:,} trainable parameters")

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config, config['training']['epochs'])

        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            device=device,
            scheduler=scheduler
        )

        # Training loop
        logger.info("Starting training...")
        best_val_score = -float('inf')
        best_epoch = 0

        for epoch in range(1, config['training']['epochs'] + 1):
            logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")

            # Train
            train_loss = trainer.train_epoch(train_loader, epoch)
            logger.info(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_loss, val_metrics = trainer.validate(val_loader)
            logger.info(f"Val Loss: {val_loss:.4f}")

            for metric_name, metric_value in val_metrics.items():
                logger.info(f"Val {metric_name}: {metric_value:.4f}")

            # Track metrics
            trainer.train_losses.append(train_loss)
            trainer.val_losses.append(val_loss)
            trainer.val_metrics.append(val_metrics)

            # Log to MLflow
            if use_mlflow:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    }, step=epoch)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('roc_auc', 0.0))
                else:
                    scheduler.step()

            # Save best model
            current_score = val_metrics.get('roc_auc', 0.0)
            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch

                model_save_path = checkpoint_dir / config['paths']['model_save_name']
                trainer.save_checkpoint(
                    str(model_save_path),
                    epoch,
                    val_metrics
                )
                logger.info(f"Best model saved with score: {best_val_score:.4f}")

            # Early stopping
            if trainer.check_early_stopping(current_score):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(f"\nTraining completed. Best validation score: {best_val_score:.4f} at epoch {best_epoch}")

        # Plot training curves
        plot_path = results_dir / "training_curves.png"
        plot_training_curves(
            trainer.train_losses,
            trainer.val_losses,
            trainer.val_metrics,
            save_path=str(plot_path)
        )

        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        test_loss, test_metrics = trainer.validate(test_loader)

        logger.info(f"Test Loss: {test_loss:.4f}")
        for metric_name, metric_value in test_metrics.items():
            logger.info(f"Test {metric_name}: {metric_value:.4f}")

        # Save results
        results = {
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'best_val_score': best_val_score,
            'best_epoch': best_epoch,
            'total_epochs': epoch,
            'num_parameters': num_params
        }

        results_path = results_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

        # Save results summary
        summary_path = results_dir / "results_summary.txt"
        save_results_summary(test_metrics, config, str(summary_path))

        # Save config
        config_save_path = results_dir / "config.yaml"
        save_config(config, str(config_save_path))

        # Log final metrics to MLflow
        if use_mlflow:
            try:
                import mlflow
                mlflow.log_metrics({
                    'test_loss': test_loss,
                    **{f'test_{k}': v for k, v in test_metrics.items()}
                })
                mlflow.log_artifact(str(plot_path))
                mlflow.log_artifact(str(results_path))
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow finalization failed: {e}")

        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        if use_mlflow:
            try:
                import mlflow
                mlflow.end_run(status='FAILED')
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
