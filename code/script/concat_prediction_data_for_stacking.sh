#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./concat_prediction_data_for_stacking.sh --target <target> --num_files <num_files> --prediction_data_folder <folder>"
  echo ""
  echo "Arguments:"
  echo "  --target                  Target dataset name (e.g., I, M, O, C)"
  echo "  --num_files               Number of prediction files to merge (e.g., 4)"
  echo "  --prediction_data_folder  Directory containing prediction pickle files"
  echo ""
  echo "Example:"
  echo "  ./concat_prediction_data_for_stacking.sh --target M --num_files 4 --prediction_data_folder /home/user/experiment/predictions"
  exit 0
fi

python ../stacking/concat_prediction_pickle_files.py "$@"