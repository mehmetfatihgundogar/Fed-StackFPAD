#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage:"
  echo "  ./prepare_datasets.sh --base_folder <folder> --num_of_frame_steps <int> --training_datasets <ids> --test_dataset <id> [--is_centralized_setup]"
  echo ""
  echo "Arguments:"
  echo "  --base_folder          Base folder path for datasets."
  echo "  --num_of_frame_steps   Number of frame steps."
  echo "  --training_datasets    Comma-separated dataset IDs (must be 1 or 4 values)."
  echo "  --test_dataset         Single dataset ID for testing."
  echo "  --is_centralized_setup Flag for centralized setup (optional)."
  echo ""
  echo "Dataset IDs:"
  echo "  1: ReplayAttack"
  echo "  2: MsuMfsd"
  echo "  3: OuluNpu"
  echo "  4: CasiaFasd"
  echo ""
  echo "Examples:"
  echo "  ./prepare_datasets.sh --base_folder /ssd/datasets --num_of_frame_steps 5 --training_datasets 1 --test_dataset 2"
  echo ""
  echo "  ./prepare_datasets.sh --base_folder /ssd/datasets --num_of_frame_steps 5 --training_datasets 1,3,4 --test_dataset 2 --is_centralized_setup"
  exit 0
fi

python ../dataset_preprocessing/preprocess_datasets.py "$@"