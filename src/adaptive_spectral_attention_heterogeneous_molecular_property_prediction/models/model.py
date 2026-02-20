"""Main model architecture: Adaptive Spectral Attention for Molecular Property Prediction."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.data.preprocessing import (
    compute_edge_types,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.models.components import (
    AdaptiveSpectralAttention,
    HeterogeneousMessagePassing,
    SpectralFilterLayer,
)

logger = logging.getLogger(__name__)


class AdaptiveSpectralAttentionModel(nn.Module):
    """
    Adaptive Spectral Attention model for molecular property prediction.

    This model combines:
    1. Heterogeneous message passing for different bond types
    2. Adaptive spectral filtering in the frequency domain
    3. Multi-head attention with spectral awareness
    4. Curriculum learning based on spectral complexity
    """

    def __init__(
        self,
        num_node_features: int,
        num_tasks: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        spectral_heads: int = 4,
        num_eigenvalues: int = 20,
        use_adaptive_filter: bool = True,
        use_heterogeneous: bool = True,
        edge_types: int = 4,
        pool_type: str = "global_mean",
        use_residual: bool = True,
        num_frequency_bands: int = 5
    ):
        """
        Initialize the model.

        Args:
            num_node_features: Number of input node features.
            num_tasks: Number of prediction tasks.
            hidden_dim: Hidden dimension size.
            num_layers: Number of graph layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            spectral_heads: Number of spectral attention heads.
            num_eigenvalues: Number of eigenvalues to use.
            use_adaptive_filter: Whether to use adaptive spectral filtering.
            use_heterogeneous: Whether to use heterogeneous message passing.
            edge_types: Number of edge types for heterogeneous graphs.
            pool_type: Type of graph pooling ('global_mean', 'global_max', 'global_add').
            use_residual: Whether to use residual connections.
            num_frequency_bands: Number of frequency bands for spectral filtering.
        """
        super().__init__()

        self.num_node_features = num_node_features
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_adaptive_filter = use_adaptive_filter
        self.use_heterogeneous = use_heterogeneous
        self.edge_types = edge_types
        self.pool_type = pool_type
        self.use_residual = use_residual

        # Input embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)

        # Graph layers
        self.graph_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            # Heterogeneous or homogeneous message passing
            if use_heterogeneous:
                graph_layer = HeterogeneousMessagePassing(
                    hidden_dim,
                    hidden_dim,
                    num_edge_types=edge_types
                )
            else:
                # Standard GNN layer (simplified)
                graph_layer = nn.Linear(hidden_dim, hidden_dim)

            self.graph_layers.append(graph_layer)

            # Adaptive spectral attention
            if use_adaptive_filter:
                attn_layer = AdaptiveSpectralAttention(
                    hidden_dim,
                    num_heads=spectral_heads,
                    num_frequency_bands=num_frequency_bands,
                    dropout=dropout
                )
            else:
                # Standard attention (simplified)
                attn_layer = nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True
                )

            self.attention_layers.append(attn_layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Pooling and prediction head
        self.dropout = nn.Dropout(dropout)

        # MLP for final prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(
        self,
        data: Batch
    ) -> Tensor:
        """
        Forward pass.

        Args:
            data: Batched PyTorch Geometric Data object.

        Returns:
            Predictions of shape [batch_size, num_tasks].
        """
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        # Get spectral features
        # Note: For batched graphs, spectral features are challenging to handle
        # We'll use a simplified approach - set them to None and let the spectral
        # filter fall back to simple transformations
        num_nodes = x.size(0)
        if hasattr(data, 'eigenvectors') and hasattr(data, 'eigenvalues'):
            eigenvectors = data.eigenvectors
            eigenvalues = data.eigenvalues
            # Check if shapes are compatible (single graph vs batched)
            if eigenvectors.size(0) != num_nodes:
                # Batched graphs - spectral features won't work directly
                eigenvectors = None
                eigenvalues = None
        else:
            # No spectral features available
            eigenvectors = None
            eigenvalues = None

        # Compute edge types if using heterogeneous
        if self.use_heterogeneous:
            if hasattr(data, 'edge_type'):
                edge_type = data.edge_type
            else:
                edge_type = compute_edge_types(data, self.edge_types)
                edge_type = edge_type.to(x.device)
        else:
            edge_type = None

        # Input embedding
        x = self.node_embedding(x)

        # Apply graph layers
        for layer_idx in range(self.num_layers):
            x_input = x

            # Message passing
            if self.use_heterogeneous:
                x = self.graph_layers[layer_idx](x, edge_index, edge_type)
            else:
                x = self.graph_layers[layer_idx](x)

            # Spectral attention
            if self.use_adaptive_filter:
                x = self.attention_layers[layer_idx](
                    x,
                    edge_index,
                    eigenvectors,
                    eigenvalues,
                    batch
                )
            else:
                # Standard self-attention (simplified, using all nodes)
                x_attn = x.unsqueeze(0)  # Add batch dim for attention
                x_attn, _ = self.attention_layers[layer_idx](
                    x_attn, x_attn, x_attn
                )
                x = x_attn.squeeze(0)

            # Residual connection
            if self.use_residual:
                x = x + x_input

            # Layer norm
            x = self.layer_norms[layer_idx](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Graph-level pooling
        if self.pool_type == "global_mean":
            x = global_mean_pool(x, batch)
        elif self.pool_type == "global_max":
            x = global_max_pool(x, batch)
        elif self.pool_type == "global_add":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Final prediction
        out = self.predictor(x)

        return out

    def get_complexity_aware_weight(
        self,
        data: Batch,
        curriculum_threshold: float = 1.0
    ) -> Tensor:
        """
        Compute sample weights based on spectral complexity for curriculum learning.

        Args:
            data: Batched PyTorch Geometric Data object.
            curriculum_threshold: Maximum complexity to include (0.0 to 1.0).

        Returns:
            Sample weights of shape [batch_size].
        """
        if hasattr(data, 'spectral_complexity'):
            complexities = data.spectral_complexity

            # Weight samples below threshold
            weights = (complexities <= curriculum_threshold).float()

            # Soft weighting based on how far below threshold
            if curriculum_threshold < 1.0:
                soft_weights = torch.clamp(
                    1.0 - (complexities / curriculum_threshold),
                    min=0.0,
                    max=1.0
                )
                weights = weights * soft_weights

            return weights
        else:
            # Uniform weights if complexity not available
            batch_size = data.batch.max().item() + 1
            return torch.ones(batch_size, device=data.batch.device)
