#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./generate_tsne_for_all_domains.sh --target_domain <domain> --feature_dir <dir> --split_type <split> --train_set <train_set> --output_file <file.pdf>"
  echo ""
  echo "Arguments:"
  echo "  --target_domain   Target domain label (e.g., M, I, C, O)"
  echo "  --feature_dir     Directory containing .npy files"
  echo "  --split_type      Split type (e.g., federated, central)"
  echo "  --train_set       Federated training set identifier (e.g., IOC, MOC, IMO, IMC)"
  echo "  --output_file     Output PDF filename"
  echo ""
  echo "Example:"
  echo "  ./generate_tsne_for_all_domains.sh --target_domain M --feature_dir ./data --split_type federated --train_set IOC --output_file tsne_federated_IOC_M.pdf"
  exit 0
fi

python ../tsne/generate_tsne_for_all_domains.py "$@"