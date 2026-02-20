#!/usr/bin/env python
"""Prediction script for inference on new molecules."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.models.model import (
    AdaptiveSpectralAttentionModel,
)
from adaptive_spectral_attention_heterogeneous_molecular_property_prediction.data.preprocessing import (
    add_spectral_features,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def smiles_to_graph(smiles: str) -> Data:
    """
    Convert SMILES string to PyTorch Geometric graph.

    Args:
        smiles: SMILES string representation of molecule.

    Returns:
        PyTorch Geometric Data object.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs(),
                atom.IsInRing(),
                atom.GetChiralTag(),
                0  # Padding to match MoleculeNet features
            ]
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Get edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Undirected graph

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # Handle molecules with no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = x.size(0)

        return data

    except Exception as e:
        logger.error(f"Error converting SMILES to graph: {e}")
        raise


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Tuple of (model, config).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Reconstruct model
    model = AdaptiveSpectralAttentionModel(
        num_node_features=config['model'].get('num_node_features', 9),
        num_tasks=config['model'].get('num_tasks', 1),
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        spectral_heads=config['model']['spectral_heads'],
        num_eigenvalues=config['model']['num_eigenvalues'],
        use_adaptive_filter=config['model']['use_adaptive_filter'],
        use_heterogeneous=config['model']['use_heterogeneous'],
        edge_types=config['model']['edge_types'],
        pool_type=config['model']['pool_type'],
        use_residual=config['model']['use_residual'],
        num_frequency_bands=config['spectral'].get('frequency_bands', 5)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model, config


def predict(
    model: torch.nn.Module,
    smiles_list: List[str],
    device: torch.device,
    num_eigenvalues: int = 20
) -> List[Dict]:
    """
    Make predictions for a list of SMILES strings.

    Args:
        model: Trained model.
        smiles_list: List of SMILES strings.
        device: Device to run inference on.
        num_eigenvalues: Number of eigenvalues for spectral features.

    Returns:
        List of prediction dictionaries.
    """
    model.eval()

    results = []

    with torch.no_grad():
        for smiles in smiles_list:
            try:
                # Convert to graph
                data = smiles_to_graph(smiles)

                # Add spectral features
                data = add_spectral_features(data, num_eigenvalues)

                # Create batch
                batch = Batch.from_data_list([data]).to(device)

                # Predict
                output = model(batch)
                prob = torch.sigmoid(output).cpu().numpy()[0]

                results.append({
                    'smiles': smiles,
                    'predictions': prob.tolist(),
                    'success': True,
                    'error': None
                })

            except Exception as e:
                logger.warning(f"Prediction failed for {smiles}: {e}")
                results.append({
                    'smiles': smiles,
                    'predictions': None,
                    'success': False,
                    'error': str(e)
                })

    return results


def main() -> None:
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict molecular properties")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--smiles',
        type=str,
        nargs='+',
        help='SMILES strings to predict (space-separated)'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='Input file with SMILES (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output file for predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Get SMILES input
    smiles_list = []

    if args.smiles:
        smiles_list.extend(args.smiles)

    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)

        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    smiles_list.append(line)

    if not smiles_list:
        logger.error("No SMILES provided. Use --smiles or --input-file")
        sys.exit(1)

    logger.info(f"Loaded {len(smiles_list)} SMILES for prediction")

    try:
        # Load model
        model, config = load_model(args.checkpoint, device)

        # Make predictions
        logger.info("Making predictions...")
        num_eigenvalues = config['model'].get('num_eigenvalues', 20)
        results = predict(model, smiles_list, device, num_eigenvalues)

        # Print results
        logger.info("\nPrediction Results:")
        logger.info("=" * 80)

        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. SMILES: {result['smiles']}")
            if result['success']:
                preds = result['predictions']
                if isinstance(preds, list):
                    for task_idx, pred in enumerate(preds):
                        logger.info(f"   Task {task_idx}: {pred:.4f} (probability)")
                else:
                    logger.info(f"   Prediction: {preds:.4f} (probability)")
            else:
                logger.info(f"   Error: {result['error']}")

        logger.info("\n" + "=" * 80)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

        # Summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"\nSummary: {successful}/{len(results)} predictions successful")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
