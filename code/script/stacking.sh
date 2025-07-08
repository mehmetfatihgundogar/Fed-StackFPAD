#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./stacking.sh --target <target> --stacked_model_type <model_type> --train_file <validation.pkl> --test_file <test.pkl> --output_model_file <model.pkl>"
  echo ""
  echo "Arguments:"
  echo "  --target               Target dataset name (e.g., I, M, O, C)"
  echo "  --stacked_model_type   One of: Svm, LassoRegression, RandomForest, XGBoost, LogisticRegression"
  echo "  --train_file           Path to validation feature pickle"
  echo "  --test_file            Path to test feature pickle"
  echo "  --output_roc_file      Path to save the data for roc curve"
  echo "  --output_model_file    Path to save trained model"
  echo ""
  echo "Example:"
  echo "  ./stacking.sh --target M --stacked_model_type Svm --train_file ./validation.pkl --test_file ./test.pkl --output_roc_file ./stacked_M.npz --output_model_file ./svm_model_M.pkl"
  exit 0
fi

python ../stacking/stacked_model_training_and_evaluation.py "$@"