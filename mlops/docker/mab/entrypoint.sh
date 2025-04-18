#!/bin/bash
set -e
echo "Verifying dependencies..."
python -c "import pandas, openpyxl, sklearn, mlflow; print('Dependencies verified.')"
echo "Starting training..."
python train_mab.py