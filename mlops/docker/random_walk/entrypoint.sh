#!/bin/bash
set -e
echo "Verifying dependencies..."
python -c "import pandas, openpyxl, networkx, mlflow; print('Dependencies verified.')"
echo "Starting training..."
python train_random_walk.py