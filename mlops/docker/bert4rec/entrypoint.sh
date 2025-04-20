#!/bin/bash
set -e
echo "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "Verifying dependencies..."
python -c "import pandas, openpyxl, transformers, mlflow, sklearn; print('Dependencies verified.')"
echo "Starting training..."
python train_bert4rec.py