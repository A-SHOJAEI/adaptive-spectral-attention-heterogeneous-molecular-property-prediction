"""
Adaptive Spectral Attention for Heterogeneous Molecular Property Prediction.

This package implements a novel graph neural network architecture that combines
adaptive spectral filtering with heterogeneous message passing for molecular
property prediction.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.models.model import (
    AdaptiveSpectralAttentionModel,
)

__all__ = ["AdaptiveSpectralAttentionModel"]
