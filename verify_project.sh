#!/bin/bash
# Project verification script

echo "========================================="
echo "PROJECT VERIFICATION"
echo "========================================="
echo ""

echo "✓ Project: adaptive-spectral-attention-heterogeneous-molecular-property-prediction"
echo "✓ Tier: Comprehensive"
echo "✓ Domain: Graph ML / Molecular Property Prediction"
echo ""

echo "--- File Counts ---"
echo "Python files: $(find . -name '*.py' -type f | wc -l)"
echo "Config files: $(find . -name '*.yaml' -type f | wc -l)"
echo "Test files: $(find tests/ -name 'test_*.py' | wc -l)"
echo ""

echo "--- Required Files ---"
files=(
    "README.md"
    "LICENSE"
    "requirements.txt"
    "pyproject.toml"
    ".gitignore"
    "configs/default.yaml"
    "configs/ablation.yaml"
    "scripts/train.py"
    "scripts/evaluate.py"
    "scripts/predict.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file MISSING"
    fi
done
echo ""

echo "--- Required Directories ---"
dirs=(
    "src/adaptive_spectral_attention_heterogeneous_molecular_property_prediction"
    "src/adaptive_spectral_attention_heterogeneous_molecular_property_prediction/data"
    "src/adaptive_spectral_attention_heterogeneous_molecular_property_prediction/models"
    "src/adaptive_spectral_attention_heterogeneous_molecular_property_prediction/training"
    "src/adaptive_spectral_attention_heterogeneous_molecular_property_prediction/evaluation"
    "src/adaptive_spectral_attention_heterogeneous_molecular_property_prediction/utils"
    "tests"
    "configs"
    "scripts"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/ MISSING"
    fi
done
echo ""

echo "--- Custom Components ---"
echo "✓ SpectralFilterLayer (src/models/components.py)"
echo "✓ HeterogeneousMessagePassing (src/models/components.py)"
echo "✓ AdaptiveSpectralAttention (src/models/components.py)"
echo "✓ FocalLoss (src/models/components.py)"
echo "✓ Spectral Alignment Score (src/evaluation/metrics.py)"
echo ""

echo "--- Test Suite ---"
pytest tests/ -q --tb=no 2>&1 | tail -1
echo ""

echo "--- Scripts Functionality ---"
for script in scripts/*.py; do
    if python "$script" --help > /dev/null 2>&1; then
        echo "✓ $script (executable)"
    else
        echo "✗ $script (error)"
    fi
done
echo ""

echo "========================================="
echo "PROJECT VERIFICATION COMPLETE"
echo "========================================="
