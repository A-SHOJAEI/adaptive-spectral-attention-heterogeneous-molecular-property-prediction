"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.data.preprocessing import (
    compute_graph_laplacian,
    compute_spectral_complexity,
    add_spectral_features,
    compute_edge_types,
)


def test_compute_graph_laplacian():
    """Test Laplacian computation."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3

    lap_edge_index, lap_edge_weight = compute_graph_laplacian(
        edge_index, num_nodes, normalization="sym"
    )

    assert lap_edge_index.size(0) == 2
    assert lap_edge_weight is not None
    assert len(lap_edge_weight) > 0


def test_compute_spectral_complexity(sample_graph):
    """Test spectral complexity computation."""
    complexity = compute_spectral_complexity(sample_graph, num_eigenvalues=5)

    assert isinstance(complexity, float)
    assert 0.0 <= complexity <= 1.0


def test_add_spectral_features(sample_graph):
    """Test adding spectral features to graph."""
    num_eigs = 5
    data_with_spectral = add_spectral_features(sample_graph, num_eigs)

    assert hasattr(data_with_spectral, 'eigenvalues')
    assert hasattr(data_with_spectral, 'eigenvectors')
    assert hasattr(data_with_spectral, 'spectral_complexity')

    assert data_with_spectral.eigenvalues.size(0) == num_eigs
    assert data_with_spectral.eigenvectors.size(1) == num_eigs


def test_compute_edge_types(sample_graph):
    """Test edge type computation."""
    edge_types = compute_edge_types(sample_graph, num_edge_types=4)

    assert edge_types.size(0) == sample_graph.edge_index.size(1)
    assert edge_types.min() >= 0
    assert edge_types.max() < 4


def test_spectral_features_edge_cases():
    """Test spectral features with edge cases."""
    # Single node graph
    data = Data(
        x=torch.randn(1, 5),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        y=torch.tensor([[1.0]])
    )
    data.num_nodes = 1

    data_with_spectral = add_spectral_features(data, num_eigenvalues=5)

    assert data_with_spectral.eigenvalues.size(0) == 5
    assert data_with_spectral.spectral_complexity == 0.0
