"""Training loop with learning rate scheduling, early stopping, and curriculum learning."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.evaluation.metrics import (
    compute_metrics,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.models.components import (
    FocalLoss,
)

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        model: Model to optimize.
        config: Configuration dictionary.

    Returns:
        Optimizer instance.
    """
    optimizer_name = config['training'].get('optimizer', 'adam').lower()
    lr = config['training'].get('learning_rate', 0.001)
    weight_decay = config['training'].get('weight_decay', 0.00001)

    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Created optimizer: {optimizer_name} with lr={lr}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_epochs: int
) -> Optional[Any]:
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: Optimizer to schedule.
        config: Configuration dictionary.
        num_epochs: Total number of training epochs.

    Returns:
        Scheduler instance or None.
    """
    scheduler_name = config['training'].get('scheduler', 'cosine').lower()

    if scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        logger.info("Created CosineAnnealingLR scheduler")
    elif scheduler_name == 'step':
        step_size = config['training'].get('step_size', 30)
        gamma = config['training'].get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Created StepLR scheduler with step_size={step_size}")
    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        logger.info("Created ReduceLROnPlateau scheduler")
    elif scheduler_name == 'none':
        scheduler = None
        logger.info("No scheduler used")
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


class Trainer:
    """
    Trainer class with early stopping, curriculum learning, and mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: torch.device,
        scheduler: Optional[Any] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train.
            optimizer: Optimizer instance.
            config: Configuration dictionary.
            device: Device to train on.
            scheduler: Learning rate scheduler.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler

        # Training settings
        self.epochs = config['training'].get('epochs', 100)
        self.grad_clip = config['training'].get('grad_clip', 1.0)
        self.use_mixed_precision = config['training'].get('use_mixed_precision', True)

        # Early stopping
        self.patience = config['training'].get('early_stopping_patience', 20)
        self.delta = config['training'].get('early_stopping_delta', 0.0001)
        self.best_score = -np.inf
        self.counter = 0
        self.early_stop = False

        # Curriculum learning
        self.use_curriculum = config['curriculum'].get('use_curriculum', False)
        self.initial_threshold = config['curriculum'].get('initial_complexity_threshold', 0.3)
        self.final_threshold = config['curriculum'].get('final_complexity_threshold', 1.0)
        self.curriculum_epochs = config['curriculum'].get('complexity_increase_epochs', 50)

        # Loss function
        # Use reduction='none' when curriculum is enabled so per-sample
        # weighting can be applied correctly.
        use_focal = config['loss'].get('use_focal_loss', False)
        loss_reduction = 'none' if self.use_curriculum else 'mean'
        if use_focal:
            focal_alpha = config['loss'].get('focal_alpha', 0.25)
            focal_gamma = config['loss'].get('focal_gamma', 2.0)
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=loss_reduction)
            logger.info("Using Focal Loss")
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction=loss_reduction)
            logger.info("Using BCEWithLogitsLoss")

        # Mixed precision
        if self.use_mixed_precision:
            try:
                # Use new API if available (torch >= 2.4)
                self.scaler = torch.amp.GradScaler('cuda')
            except (AttributeError, TypeError):
                # Fall back to old API
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

    def get_curriculum_threshold(self, epoch: int) -> float:
        """
        Compute curriculum complexity threshold for current epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Complexity threshold (0.0 to 1.0).
        """
        if not self.use_curriculum:
            return 1.0

        if epoch >= self.curriculum_epochs:
            return self.final_threshold

        # Linear increase from initial to final threshold
        progress = epoch / self.curriculum_epochs
        threshold = self.initial_threshold + progress * (
            self.final_threshold - self.initial_threshold
        )

        return threshold

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        curriculum_threshold = self.get_curriculum_threshold(epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            batch = batch.to(self.device)

            # Handle missing labels
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

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.use_mixed_precision:
                try:
                    autocast_context = torch.amp.autocast('cuda')
                except (AttributeError, TypeError):
                    autocast_context = torch.cuda.amp.autocast()
                with autocast_context:
                    outputs = self.model(batch)
                    outputs = outputs[valid_mask]
                    labels = labels[valid_mask]
                    loss = self.criterion(outputs, labels)

                    # Apply curriculum weighting if enabled
                    if self.use_curriculum:
                        weights = self.model.get_complexity_aware_weight(
                            batch,
                            curriculum_threshold
                        )
                        weights = weights[valid_mask]
                        loss = (loss * weights.unsqueeze(-1)).mean()

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                outputs = outputs[valid_mask]
                labels = labels[valid_mask]
                loss = self.criterion(outputs, labels)

                # Apply curriculum weighting if enabled
                if self.use_curriculum:
                    weights = self.model.get_complexity_aware_weight(
                        batch,
                        curriculum_threshold
                    )
                    weights = weights[valid_mask]
                    loss = (loss * weights.unsqueeze(-1)).mean()

                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip
                    )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (average loss, metrics dictionary).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

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

                outputs = self.model(batch)
                outputs = outputs[valid_mask]
                labels = labels[valid_mask]

                loss = self.criterion(outputs, labels)
                # Always reduce to scalar for validation loss tracking
                if loss.dim() > 0:
                    loss = loss.mean()
                total_loss += loss.item()
                num_batches += 1

                # Collect predictions
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        if all_preds:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            metrics = compute_metrics(all_labels, all_preds)
        else:
            metrics = {}

        return avg_loss, metrics

    def check_early_stopping(self, val_score: float) -> bool:
        """
        Check early stopping criterion.

        Args:
            val_score: Validation score (higher is better).

        Returns:
            True if training should stop, False otherwise.
        """
        if val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                return True
            return False

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            metrics: Validation metrics.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
