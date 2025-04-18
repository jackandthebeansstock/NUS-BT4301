#!/bin/bash
set -e
echo "Verifying dependencies..."
python -c "import pandas, openpyxl, implicit, mlflow; print('Dependencies verified.')"
echo "Starting training..."
python train_als.py