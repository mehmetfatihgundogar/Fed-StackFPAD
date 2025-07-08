#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./prepare_vit_model_based_on_imagenet22K_encoder_weights.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --root_folder       root folder containing the model files"
  echo "  --imagenet_file     filename of MAE model pretrained on ImageNet-22k"
  echo "  --base_vit_model_file          filename of base ViT model"
  echo "  --output_file       file name of the output model (ViT model with ImageNet-22k encoder weights)"
  exit 0
fi
# This script automates the merging of ImageNet-22K encoder weights into an base ViT model.

# Sample Usage:
# ./prepare_vit_model_based_on_imagenet22K_encoder_weights.sh --root_folder /path/to/models --imagenet_file mae_pretrain_vit_base.pth --base_vit_model_file base_vit_model.pth --output_file vit_model_with_imagenet22k_encoder_weights.pth

python ../imagenet_utils/merge_imagenet_weights_to_mae_vit_model.py "$@"
