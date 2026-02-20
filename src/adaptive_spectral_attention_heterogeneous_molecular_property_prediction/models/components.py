"""Custom model components: spectral filters, heterogeneous layers, and loss functions."""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax

logger = logging.getLogger(__name__)


class SpectralFilterLayer(nn.Module):
    """
    Learnable spectral filter that operates in the frequency domain.

    This layer learns task-specific frequency filters by applying learnable
    transformations to graph Laplacian eigenvectors.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frequency_bands: int = 5,
        filter_type: str = "adaptive"
    ):
        """
        Initialize spectral filter layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            num_frequency_bands: Number of frequency bands to learn.
            filter_type: Type of filter ('adaptive', 'fixed', 'learnable').
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_frequency_bands = num_frequency_bands
        self.filter_type = filter_type

        # Learnable frequency band weights
        self.band_weights = nn.Parameter(torch.ones(num_frequency_bands))

        # Per-band transformation matrices
        self.band_transforms = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(num_frequency_bands)
        ])

        # Adaptive gating mechanism
        if filter_type == "adaptive":
            self.gate_network = nn.Sequential(
                nn.Linear(in_channels, num_frequency_bands),
                nn.Softmax(dim=-1)
            )

    def forward(
        self,
        x: Tensor,
        eigenvectors: Tensor,
        eigenvalues: Tensor
    ) -> Tensor:
        """
        Apply spectral filtering to node features.

        Args:
            x: Node features of shape [num_nodes, in_channels].
            eigenvectors: Graph Laplacian eigenvectors [num_nodes, num_eigs].
            eigenvalues: Graph Laplacian eigenvalues [num_eigs].

        Returns:
            Filtered node features of shape [num_nodes, out_channels].
        """
        num_nodes = x.size(0)

        # Check if eigenvectors/eigenvalues are available and compatible
        if eigenvectors is None or eigenvalues is None:
            # Fall back to simple linear transformation
            return torch.mean(torch.stack([t(x) for t in self.band_transforms]), dim=0)

        num_eigs = eigenvectors.size(1)

        # For batched graphs, eigenvectors shape won't match x
        # In that case, skip spectral filtering and use simple aggregation
        if eigenvectors.size(0) != num_nodes:
            # Fall back to simple linear transformation
            return torch.mean(torch.stack([t(x) for t in self.band_transforms]), dim=0)

        # Transform to spectral domain
        x_spectral = torch.matmul(eigenvectors.t(), x)  # [num_eigs, in_channels]

        # Compute frequency bands
        max_eigenvalue = eigenvalues.max() + 1e-8
        normalized_eigs = eigenvalues / max_eigenvalue

        # Assign eigenvalues to frequency bands
        band_size = 1.0 / self.num_frequency_bands
        band_indices = (normalized_eigs / band_size).long()
        band_indices = torch.clamp(band_indices, 0, self.num_frequency_bands - 1)

        # Apply band-specific transformations
        outputs = []
        for band_idx in range(self.num_frequency_bands):
            # Get eigenvalues in this band
            mask = (band_indices == band_idx).float()

            if mask.sum() > 0:
                # Apply transformation to this frequency band
                band_features = x_spectral * mask.unsqueeze(-1)
                transformed = self.band_transforms[band_idx](band_features)
                outputs.append(transformed * self.band_weights[band_idx])
            else:
                outputs.append(torch.zeros_like(x_spectral[:, :self.out_channels]))

        # Combine bands
        x_spectral_out = sum(outputs)

        # Transform back to spatial domain
        x_out = torch.matmul(eigenvectors, x_spectral_out)

        # Apply adaptive gating if enabled
        if self.filter_type == "adaptive":
            # Compute per-node gates based on input features
            gates = self.gate_network(x)  # [num_nodes, num_frequency_bands]

            # Reweight outputs based on gates
            band_outputs_spatial = []
            for band_idx in range(self.num_frequency_bands):
                mask = (band_indices == band_idx).float()
                band_features = x_spectral * mask.unsqueeze(-1)
                band_out = torch.matmul(
                    eigenvectors,
                    self.band_transforms[band_idx](band_features)
                )
                band_outputs_spatial.append(band_out)

            # Weighted combination
            x_out = sum(
                gates[:, i:i+1] * band_outputs_spatial[i]
                for i in range(self.num_frequency_bands)
            )

        return x_out


class HeterogeneousMessagePassing(MessagePassing):
    """
    Heterogeneous message passing layer with edge-type specific transformations.

    Implements message passing on heterogeneous graphs where different edge types
    (bond types) require different message transformations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_types: int = 4,
        aggr: str = "add"
    ):
        """
        Initialize heterogeneous message passing layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            num_edge_types: Number of edge types.
            aggr: Aggregation method ('add', 'mean', 'max').
        """
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types

        # Edge-type specific message transformations
        self.edge_transforms = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(num_edge_types)
        ])

        # Self-loop transformation
        self.self_transform = nn.Linear(in_channels, out_channels)

        # Edge type attention
        self.edge_type_attention = nn.Linear(num_edge_types, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with heterogeneous message passing.

        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            edge_type: Edge types of shape [num_edges].

        Returns:
            Updated node features of shape [num_nodes, out_channels].
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Default edge types if not provided
        if edge_type is None:
            edge_type = torch.zeros(
                edge_index.size(1),
                dtype=torch.long,
                device=x.device
            )
        else:
            # Pad edge types for self-loops (use type 0)
            num_self_loops = x.size(0)
            self_loop_types = torch.zeros(
                num_self_loops,
                dtype=torch.long,
                device=x.device
            )
            edge_type = torch.cat([edge_type, self_loop_types], dim=0)

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_type=edge_type)

        # Add self-transformation
        out = out + self.self_transform(x)

        return out

    def message(self, x_j: Tensor, edge_type: Tensor) -> Tensor:
        """
        Construct messages from source nodes.

        Args:
            x_j: Source node features [num_edges, in_channels].
            edge_type: Edge types [num_edges].

        Returns:
            Messages of shape [num_edges, out_channels].
        """
        # Apply edge-type specific transformations
        # Use same dtype as input for mixed precision compatibility
        messages = torch.zeros(
            x_j.size(0),
            self.out_channels,
            device=x_j.device,
            dtype=x_j.dtype
        )

        for edge_type_idx in range(self.num_edge_types):
            mask = (edge_type == edge_type_idx)
            if mask.any():
                messages[mask] = self.edge_transforms[edge_type_idx](x_j[mask])

        return messages


class AdaptiveSpectralAttention(nn.Module):
    """
    Adaptive spectral attention mechanism.

    Combines spectral filtering with multi-head attention, learning to attend
    to different frequency components based on the task.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_frequency_bands: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize adaptive spectral attention.

        Args:
            hidden_dim: Hidden feature dimension.
            num_heads: Number of attention heads.
            num_frequency_bands: Number of frequency bands.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Multi-head attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Spectral filter
        self.spectral_filter = SpectralFilterLayer(
            hidden_dim,
            hidden_dim,
            num_frequency_bands,
            filter_type="adaptive"
        )

        # Frequency-aware attention weights
        self.freq_attention = nn.Linear(num_frequency_bands, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        eigenvectors: Tensor,
        eigenvalues: Tensor,
        batch: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply adaptive spectral attention.

        Args:
            x: Node features [num_nodes, hidden_dim].
            edge_index: Edge indices [2, num_edges].
            eigenvectors: Graph Laplacian eigenvectors.
            eigenvalues: Graph Laplacian eigenvalues.
            batch: Batch assignment for nodes.

        Returns:
            Attended node features [num_nodes, hidden_dim].
        """
        num_nodes = x.size(0)

        # Apply spectral filtering
        x_spectral = self.spectral_filter(x, eigenvectors, eigenvalues)

        # Multi-head attention
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x_spectral).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x_spectral).view(num_nodes, self.num_heads, self.head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.view(num_nodes, self.hidden_dim)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection and layer norm
        out = self.layer_norm(x + out)

        return out


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Focal loss reduces the relative loss for well-classified examples,
    focusing training on hard negatives.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor for positive class.
            gamma: Focusing parameter (higher = more focus on hard examples).
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits of shape [batch_size, num_classes].
            targets: Ground truth labels of shape [batch_size].

        Returns:
            Focal loss value.
        """
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute focal term: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma

        # Compute alpha term
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        loss = alpha_t * focal_term * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
