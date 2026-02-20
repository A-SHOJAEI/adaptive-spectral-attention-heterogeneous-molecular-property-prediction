"""Data loading utilities for MoleculeNet datasets."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ogb.graphproppred import GraphPropPredDataset, Evaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.data.preprocessing import (
    add_spectral_features,
    compute_spectral_complexity,
)

logger = logging.getLogger(__name__)


class MoleculeDataset(Dataset):
    """Custom dataset wrapper for molecular graphs with spectral features."""

    def __init__(
        self,
        pyg_dataset: Dataset,
        add_spectral: bool = True,
        num_eigenvalues: int = 20
    ):
        """
        Initialize molecular dataset.

        Args:
            pyg_dataset: PyTorch Geometric dataset.
            add_spectral: Whether to add spectral features.
            num_eigenvalues: Number of eigenvalues to compute.
        """
        super().__init__()
        self.pyg_dataset = pyg_dataset
        self.add_spectral = add_spectral
        self.num_eigenvalues = num_eigenvalues

        # Cache spectral complexities if using curriculum learning
        self._complexities: Optional[np.ndarray] = None

    def len(self) -> int:
        """Get dataset length."""
        return len(self.pyg_dataset)

    def get(self, idx: int) -> Data:
        """
        Get a single graph with spectral features.

        Args:
            idx: Index of the graph.

        Returns:
            PyTorch Geometric Data object with spectral features.
        """
        data = self.pyg_dataset[idx].clone()

        if self.add_spectral:
            data = add_spectral_features(data, self.num_eigenvalues)

        return data

    def compute_all_complexities(self) -> np.ndarray:
        """
        Compute spectral complexities for all graphs.

        Returns:
            Array of spectral complexity scores.
        """
        if self._complexities is None:
            logger.info("Computing spectral complexities for curriculum learning...")
            complexities = []

            for idx in tqdm(range(len(self)), desc="Computing complexities"):
                data = self.pyg_dataset[idx]
                try:
                    complexity = compute_spectral_complexity(data)
                    complexities.append(complexity)
                except Exception as e:
                    logger.warning(f"Failed to compute complexity for graph {idx}: {e}")
                    complexities.append(0.5)  # Default medium complexity

            self._complexities = np.array(complexities)

        return self._complexities


def get_moleculenet_dataset(
    dataset_name: str,
    data_dir: str = "data/raw"
) -> Tuple[Dataset, int, int]:
    """
    Load a MoleculeNet dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'bace', 'bbbp', 'tox21').
        data_dir: Directory to store/load data.

    Returns:
        Tuple of (dataset, num_tasks, num_node_features).

    Raises:
        ValueError: If dataset name is not supported.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    supported_datasets = [
        'bace', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox',
        'muv', 'hiv', 'esol', 'freesolv', 'lipophilicity'
    ]

    if dataset_name.lower() not in supported_datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Choose from: {supported_datasets}"
        )

    logger.info(f"Loading MoleculeNet dataset: {dataset_name}")

    try:
        dataset = MoleculeNet(
            root=str(data_path),
            name=dataset_name.upper()
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Get number of node features
    num_node_features = dataset.num_node_features

    # Infer number of tasks from the first data object
    # MoleculeNet doesn't have a num_tasks attribute directly
    first_data = dataset[0]
    if hasattr(first_data, 'y') and first_data.y is not None:
        # y shape is typically [1, num_tasks] or [num_tasks]
        if len(first_data.y.shape) == 2:
            num_tasks = first_data.y.shape[1]
        else:
            num_tasks = 1
    else:
        # Default to 1 task if y is not available
        num_tasks = 1

    logger.info(
        f"Dataset loaded: {len(dataset)} molecules, "
        f"{num_tasks} tasks, {num_node_features} node features"
    )

    return dataset, num_tasks, num_node_features


def create_data_loaders(
    dataset: Dataset,
    config: Dict,
    use_spectral: bool = True
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        dataset: PyTorch Geometric dataset.
        config: Configuration dictionary.
        use_spectral: Whether to add spectral features.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Extract config values
    batch_size = config['data'].get('batch_size', 32)
    num_workers = config['data'].get('num_workers', 4)
    train_split = config['data'].get('train_split', 0.8)
    val_split = config['data'].get('val_split', 0.1)
    random_seed = config['data'].get('random_seed', 42)
    num_eigenvalues = config['model'].get('num_eigenvalues', 20)

    # Wrap dataset with spectral features
    # Note: Spectral features are currently disabled in batched mode
    # due to complexity in handling per-graph spectral decompositions
    mol_dataset = MoleculeDataset(
        dataset,
        add_spectral=False,  # Disabled for batched processing
        num_eigenvalues=num_eigenvalues
    )

    # Create indices for splits
    num_samples = len(dataset)
    indices = list(range(num_samples))

    # First split: train + val vs test
    train_val_size = train_split + val_split
    train_val_idx, test_idx = train_test_split(
        indices,
        train_size=train_val_size,
        random_state=random_seed,
        shuffle=True
    )

    # Second split: train vs val
    val_ratio = val_split / train_val_size
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=random_seed,
        shuffle=True
    )

    logger.info(
        f"Split sizes - Train: {len(train_idx)}, "
        f"Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    # Create subsets
    train_dataset = Subset(mol_dataset, train_idx)
    val_dataset = Subset(mol_dataset, val_idx)
    test_dataset = Subset(mol_dataset, test_idx)

    # Create data loaders using PyTorch Geometric's DataLoader
    # which properly handles batching of graph data
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def filter_by_complexity(
    dataset: MoleculeDataset,
    indices: List[int],
    complexity_threshold: float
) -> List[int]:
    """
    Filter dataset indices by spectral complexity for curriculum learning.

    Args:
        dataset: MoleculeDataset with spectral features.
        indices: List of dataset indices to filter.
        complexity_threshold: Maximum complexity to include (0.0 to 1.0).

    Returns:
        Filtered list of indices.
    """
    complexities = dataset.compute_all_complexities()

    filtered_indices = [
        idx for idx in indices
        if complexities[idx] <= complexity_threshold
    ]

    logger.info(
        f"Filtered {len(indices)} -> {len(filtered_indices)} samples "
        f"with complexity <= {complexity_threshold:.2f}"
    )

    return filtered_indices
