"""Tests for training components."""

import pytest
import torch
import numpy as np

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.models.model import (
    AdaptiveSpectralAttentionModel,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.training.trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.evaluation.metrics import (
    compute_metrics,
    compute_roc_auc,
    compute_average_precision,
    compute_spectral_alignment_score,
)


def test_create_optimizer(sample_config, device):
    """Test optimizer creation."""
    model = AdaptiveSpectralAttentionModel(
        num_node_features=9,
        num_tasks=1,
        **sample_config['model']
    ).to(device)

    optimizer = create_optimizer(model, sample_config)

    assert optimizer is not None
    assert len(optimizer.param_groups) > 0


def test_create_scheduler(sample_config, device):
    """Test scheduler creation."""
    model = AdaptiveSpectralAttentionModel(
        num_node_features=9,
        num_tasks=1,
        **sample_config['model']
    ).to(device)

    optimizer = create_optimizer(model, sample_config)
    scheduler = create_scheduler(optimizer, sample_config, num_epochs=10)

    assert scheduler is not None


def test_trainer_initialization(sample_config, device):
    """Test trainer initialization."""
    model = AdaptiveSpectralAttentionModel(
        num_node_features=9,
        num_tasks=1,
        **sample_config['model']
    ).to(device)

    optimizer = create_optimizer(model, sample_config)
    scheduler = create_scheduler(optimizer, sample_config, num_epochs=10)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=sample_config,
        device=device,
        scheduler=scheduler
    )

    assert trainer is not None
    assert trainer.epochs == 2
    assert trainer.patience == 5


def test_curriculum_threshold():
    """Test curriculum learning threshold computation."""
    config = {
        'training': {'epochs': 100},
        'curriculum': {
            'use_curriculum': True,
            'initial_complexity_threshold': 0.3,
            'final_complexity_threshold': 1.0,
            'complexity_increase_epochs': 50
        },
        'loss': {'use_focal_loss': False}
    }

    model = AdaptiveSpectralAttentionModel(num_node_features=9, num_tasks=1)
    optimizer = torch.optim.Adam(model.parameters())

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=torch.device('cpu')
    )

    # Test at different epochs
    threshold_0 = trainer.get_curriculum_threshold(0)
    threshold_25 = trainer.get_curriculum_threshold(25)
    threshold_50 = trainer.get_curriculum_threshold(50)

    assert threshold_0 == 0.3
    assert 0.3 < threshold_25 < 1.0
    assert threshold_50 == 1.0


def test_compute_roc_auc(sample_labels, sample_predictions):
    """Test ROC-AUC computation."""
    roc_auc = compute_roc_auc(sample_labels, sample_predictions)

    assert 0.0 <= roc_auc <= 1.0


def test_compute_average_precision(sample_labels, sample_predictions):
    """Test average precision computation."""
    avg_prec = compute_average_precision(sample_labels, sample_predictions)

    assert 0.0 <= avg_prec <= 1.0


def test_compute_spectral_alignment_score(sample_labels, sample_predictions):
    """Test spectral alignment score."""
    complexities = np.array([0.2, 0.5, 0.7, 0.9])

    score = compute_spectral_alignment_score(
        sample_labels,
        sample_predictions,
        complexities
    )

    assert 0.0 <= score <= 1.0


def test_compute_all_metrics(sample_labels, sample_predictions):
    """Test computing all metrics."""
    metrics = compute_metrics(sample_labels, sample_predictions)

    assert 'roc_auc' in metrics
    assert 'avg_precision' in metrics
    assert 'spectral_alignment_score' in metrics
    assert 'accuracy' in metrics
    assert 'f1' in metrics

    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_early_stopping():
    """Test early stopping mechanism."""
    config = {
        'training': {
            'epochs': 100,
            'early_stopping_patience': 3,
            'early_stopping_delta': 0.001
        },
        'curriculum': {'use_curriculum': False},
        'loss': {'use_focal_loss': False}
    }

    model = AdaptiveSpectralAttentionModel(num_node_features=9, num_tasks=1)
    optimizer = torch.optim.Adam(model.parameters())

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=torch.device('cpu')
    )

    # Simulate stagnating validation scores
    assert not trainer.check_early_stopping(0.8)
    assert not trainer.check_early_stopping(0.81)  # Improvement
    assert not trainer.check_early_stopping(0.81)  # No improvement (1)
    assert not trainer.check_early_stopping(0.809)  # No improvement (2)
    assert trainer.check_early_stopping(0.808)  # No improvement (3) -> stop
