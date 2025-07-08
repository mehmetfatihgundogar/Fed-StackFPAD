#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./evaluation_of_finetuned_model.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --data_path              path to dataset (e.g.,
                                   /path/to/datasets/fed-stackfpad__exp_mode_singleclient_train_on_O__test_on_M,
                                   /path/to/datasets/fed-stackfpad__exp_mode_multiclient_train_on_IOC__test_on_M,
                                   /path/to/datasets/fed-stackfpad__exp_mode_centralized_train_on_OCI__test_on_M)"
  echo "  --data_set               dataset name for the experiment (e.g.,
                                   Fed-StackFPAD-TRAIN-IMC__TEST-O,
                                   Fed-StackFPAD-TRAIN-O__TEST-M)"
  echo "  --resume                 path to finetuned model checkpoint"
  echo "  --output_dir             output directory"
  echo "  --model                  model architecture (e.g., vit_base_patch16)"
  echo "  --batch_size             batch size"
  echo "  --split_type             split type (e.g., central, federated)"
  echo "  --n_clients              number of federated clients"
  echo "  --eval                   evaluation mode"
  echo "  --predictions_dir        directory for saving model prediction output"
  echo "  --tsne_dir               directory for saving data for tsne plot generation"
  echo "  --roc_dir                directory for saving data for roc plot generation"
  echo "  --project                project name for wandb setup"
  echo "  --wandb_key              key for logging on wandb platform"
  exit 0
fi

# Sample usage:
# ./evaluation_of_finetuned_model.sh \
# --data_path /path/to/datasets/fed-stackfpad__exp_mode_multiclient_train_on_IOC__test_on_M \
# --data_set Fed-StackFPAD-TRAIN-IOC__TEST-M \
# --resume /path/to/finetunedmodel/checkpoint-best.pth \
# --output_dir /path/to/finetunedmodel/eval_results \
# --model vit_base_patch16 \
# --batch_size 64 \
# --split_type central \
# --n_clients 3 \
# --eval \
# --predictions_dir /path/to/finetunedmodel/predictions \
# --tsne_dir /path/to/finetunedmodel/tsne \
# --roc_dir /path/to/finetunedmodel/roc_data \
# --project fed_stackfpad \
# --wandb_key your_wandb_key_here

python ../finetuning/evaluate_finetuned_model.py "$@"