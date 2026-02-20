"""Graph preprocessing and spectral feature computation."""

import logging
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    degree,
    to_scipy_sparse_matrix,
    get_laplacian,
)

logger = logging.getLogger(__name__)


def compute_graph_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalization: str = "sym"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the graph Laplacian matrix.

    Args:
        edge_index: Edge indices of shape [2, num_edges].
        num_nodes: Number of nodes in the graph.
        normalization: Type of normalization ('sym' or None).

    Returns:
        Tuple of (laplacian_edge_index, laplacian_edge_weight).
    """
    # Compute Laplacian
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index,
        normalization=normalization,
        num_nodes=num_nodes
    )

    return edge_index_lap, edge_weight_lap


def compute_eigendecomposition(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition of graph Laplacian.

    Args:
        edge_index: Laplacian edge indices.
        edge_weight: Laplacian edge weights.
        num_nodes: Number of nodes.
        k: Number of smallest eigenvalues/eigenvectors to compute.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    # Convert to scipy sparse matrix
    lap_matrix = to_scipy_sparse_matrix(
        edge_index,
        edge_weight,
        num_nodes=num_nodes
    )

    # Ensure symmetric
    lap_matrix = (lap_matrix + lap_matrix.T) / 2.0

    # Compute eigendecomposition
    k_actual = min(k, num_nodes - 2)

    if k_actual < 1:
        k_actual = 1

    try:
        eigenvalues, eigenvectors = sp.linalg.eigsh(
            lap_matrix,
            k=k_actual,
            which='SM',  # Smallest magnitude
            return_eigenvectors=True
        )
    except Exception as e:
        logger.warning(f"Eigendecomposition failed: {e}. Using zeros.")
        eigenvalues = np.zeros(k_actual)
        eigenvectors = np.zeros((num_nodes, k_actual))

    # Sort by eigenvalues
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Pad if necessary
    if k_actual < k:
        pad_eig = k - k_actual
        eigenvalues = np.pad(eigenvalues, (0, pad_eig), mode='constant')
        eigenvectors = np.pad(
            eigenvectors,
            ((0, 0), (0, pad_eig)),
            mode='constant'
        )

    return eigenvalues, eigenvectors


def compute_spectral_complexity(
    data: Data,
    num_eigenvalues: int = 20
) -> float:
    """
    Compute spectral complexity score based on eigenvalue distribution.

    The complexity is measured by the spread and magnitude of eigenvalues.
    Higher spectral gap and more distributed eigenvalues indicate higher complexity.

    Args:
        data: PyTorch Geometric Data object.
        num_eigenvalues: Number of eigenvalues to consider.

    Returns:
        Spectral complexity score between 0 and 1.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    if num_nodes < 2 or edge_index.size(1) == 0:
        return 0.0

    # Compute Laplacian
    edge_index_lap, edge_weight_lap = compute_graph_laplacian(
        edge_index,
        num_nodes,
        normalization="sym"
    )

    # Compute eigenvalues
    eigenvalues, _ = compute_eigendecomposition(
        edge_index_lap,
        edge_weight_lap,
        num_nodes,
        k=num_eigenvalues
    )

    # Complexity based on eigenvalue distribution
    # Higher variance and spectral gap indicate higher complexity
    if len(eigenvalues) > 1:
        spectral_gap = eigenvalues[1] - eigenvalues[0]
        eigenvalue_std = np.std(eigenvalues)
        max_eigenvalue = eigenvalues[-1]

        # Normalize components
        gap_score = min(spectral_gap / 2.0, 1.0)
        std_score = min(eigenvalue_std / 2.0, 1.0)
        max_score = min(max_eigenvalue / 2.0, 1.0)

        complexity = (gap_score + std_score + max_score) / 3.0
    else:
        complexity = 0.0

    return float(np.clip(complexity, 0.0, 1.0))


def add_spectral_features(
    data: Data,
    num_eigenvalues: int = 20
) -> Data:
    """
    Add spectral features to graph data.

    Args:
        data: PyTorch Geometric Data object.
        num_eigenvalues: Number of eigenvalues/eigenvectors to compute.

    Returns:
        Data object with added spectral features.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    if num_nodes < 2 or edge_index.size(1) == 0:
        # Handle edge cases
        data.eigenvalues = torch.zeros(num_eigenvalues)
        data.eigenvectors = torch.zeros(num_nodes, num_eigenvalues)
        data.spectral_complexity = torch.tensor(0.0)
        return data

    # Compute Laplacian
    edge_index_lap, edge_weight_lap = compute_graph_laplacian(
        edge_index,
        num_nodes,
        normalization="sym"
    )

    # Compute eigendecomposition
    eigenvalues, eigenvectors = compute_eigendecomposition(
        edge_index_lap,
        edge_weight_lap,
        num_nodes,
        k=num_eigenvalues
    )

    # Add to data object
    data.eigenvalues = torch.from_numpy(eigenvalues).float()
    data.eigenvectors = torch.from_numpy(eigenvectors).float()

    # Compute complexity
    complexity = compute_spectral_complexity(data, num_eigenvalues)
    data.spectral_complexity = torch.tensor(complexity).float()

    return data


def compute_edge_types(data: Data, num_edge_types: int = 4) -> torch.Tensor:
    """
    Assign edge types based on bond features or create pseudo-types.

    Args:
        data: PyTorch Geometric Data object.
        num_edge_types: Number of edge types to create.

    Returns:
        Tensor of edge types for each edge.
    """
    num_edges = data.edge_index.size(1)

    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        # Use existing edge attributes to determine types
        edge_attr = data.edge_attr

        if edge_attr.dim() == 1:
            edge_types = edge_attr.long() % num_edge_types
        else:
            # Use first feature dimension as bond type indicator.
            # Map the raw integer/float values directly via modulo to get
            # valid edge type indices in [0, num_edge_types).
            edge_types = edge_attr[:, 0].long() % num_edge_types
    else:
        # Create pseudo edge types based on node degrees
        row, col = data.edge_index
        deg_row = degree(row, num_nodes=data.num_nodes)[row]
        deg_col = degree(col, num_nodes=data.num_nodes)[col]

        # Edge type based on degree sum
        deg_sum = (deg_row + deg_col).long()
        edge_types = deg_sum % num_edge_types

    return edge_types
