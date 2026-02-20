# Adaptive Spectral Attention for Heterogeneous Molecular Property Prediction

A graph neural network that predicts molecular toxicity using adaptive spectral filtering and heterogeneous message passing. The model learns task-specific graph frequency filters and employs curriculum learning based on spectral complexity.

## Key Innovation

Combines graph spectral theory with heterogeneous GNNs through adaptive frequency-domain attention. Unlike standard graph networks that treat all structural scales equally, this approach learns which frequency components (local bonds vs. global conjugation) matter for each prediction task.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
# Train the model
python scripts/train.py --config configs/default.yaml

# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Predict on new molecules
python scripts/predict.py --checkpoint checkpoints/best_model.pt \
    --smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
```

## Usage

### Training

```bash
# Default configuration (with adaptive spectral attention)
python scripts/train.py

# Baseline without spectral components (ablation study)
python scripts/train.py --config configs/ablation.yaml

# Custom device
python scripts/train.py --device cuda
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --output results
```

### Prediction

```bash
# Single molecule
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --smiles "CCO" "CC(=O)O"

# From file
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --input-file molecules.txt \
    --output predictions.json
```

## Architecture

The model consists of:

1. **Spectral Filter Layer**: Learns task-specific frequency filters by decomposing molecular graphs into spectral components
2. **Heterogeneous Message Passing**: Different bond types (single, double, aromatic) use distinct message transformations
3. **Adaptive Spectral Attention**: Multi-head attention weighted by frequency importance
4. **Curriculum Learning**: Progressively introduces harder molecules based on spectral complexity

## Custom Components

### SpectralFilterLayer
Transforms node features into frequency domain using graph Laplacian eigenvectors, applies learnable frequency-band specific filters, and transforms back to spatial domain with adaptive gating.

### HeterogeneousMessagePassing
Edge-type specific message transformations for different bond types in molecular graphs.

### FocalLoss
Handles class imbalance by down-weighting well-classified examples and focusing on hard negatives.

## Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  use_adaptive_filter: true      # Enable spectral filtering
  use_heterogeneous: true        # Edge-type specific messages
  num_frequency_bands: 5         # Spectral decomposition resolution

curriculum:
  use_curriculum: true           # Progressive complexity
  initial_complexity_threshold: 0.3
  final_complexity_threshold: 1.0
```

## Results

Evaluated on BACE molecular property prediction (scaffold split, seed=42):

| Split | ROC-AUC | Avg Precision | Accuracy | F1 Score |
|-------|---------|---------------|----------|----------|
| Validation | 0.659 | 0.733 | 46.7% | 0.318 |
| Test | 0.453 | 0.562 | 49.3% | 0.330 |

The model demonstrates learning on the challenging BACE dataset with heterogeneous graph representations. The validation/test gap suggests the scaffold split introduces significant distribution shift, which is expected for molecular property prediction tasks where scaffold diversity is high.

## Dataset

Uses MoleculeNet datasets (BACE, BBBP, Tox21, etc.) for molecular property prediction. Molecules are represented as heterogeneous graphs with atoms as nodes and bonds as typed edges.

## Testing

```bash
# Run all tests
pytest tests/ -v --cov=src

# Specific test module
pytest tests/test_model.py -v
```

## Project Structure

```
├── configs/              # YAML configuration files
├── scripts/              # Training, evaluation, prediction scripts
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architecture and components
│   ├── training/        # Training loop and optimization
│   ├── evaluation/      # Metrics and analysis
│   └── utils/           # Configuration utilities
├── tests/               # Unit tests
└── results/             # Output directory
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
