#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./generate_roc_plot.sh --input_dir <dir> --target <target_domain> --output_dir <output_dir>"
  echo ""
  echo "Arguments:"
  echo "  --input_dir   Directory containing .npz files"
  echo "  --target      Target domain letter (e.g., I, M, O, C)"
  echo "  --output_dir  Directory to save the output PDF"
  echo ""
  echo "Example:"
  echo "  ./generate_roc_plot.sh --input_dir ./roc_data --target M --output_dir ./plots"
  exit 0
fi

python ../roc/generate_roc_plot.py "$@"