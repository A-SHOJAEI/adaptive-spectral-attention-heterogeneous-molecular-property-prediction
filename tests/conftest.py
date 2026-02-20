"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch


@pytest.fixture
def device():
    """Get computation device."""
    return torch.device('cpu')


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'dataset_name': 'bace',
            'batch_size': 4,
            'num_workers': 0,
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'random_seed': 42
        },
        'model': {
            'hidden_dim': 32,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'spectral_heads': 2,
            'num_eigenvalues': 10,
            'use_adaptive_filter': True,
            'use_heterogeneous': True,
            'edge_types': 4,
            'pool_type': 'global_mean',
            'use_residual': True
        },
        'training': {
            'epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.00001,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'grad_clip': 1.0,
            'early_stopping_patience': 5,
            'use_mixed_precision': False
        },
        'curriculum': {
            'use_curriculum': False
        },
        'spectral': {
            'use_spectral_attention': True,
            'frequency_bands': 3
        },
        'loss': {
            'use_focal_loss': False
        },
        'paths': {
            'checkpoint_dir': 'checkpoints',
            'results_dir': 'results'
        }
    }


@pytest.fixture
def sample_graph():
    """Create a sample molecular graph."""
    # Simple graph with 5 nodes
    num_nodes = 5
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)

    x = torch.randn(num_nodes, 9)  # MoleculeNet typically has 9 atom features
    y = torch.tensor([[1.0]])  # Binary label

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes

    return data


@pytest.fixture
def sample_batch(sample_graph):
    """Create a batch of sample graphs."""
    graphs = [sample_graph for _ in range(4)]
    batch = Batch.from_data_list(graphs)
    return batch


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return np.array([[1], [0], [1], [0]])


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return np.array([[0.8], [0.3], [0.7], [0.2]])
