#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./optuna_hyper_parameter_tuning.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --data_set                  dataset name for the experiment (e.g.,
                                      Fed-StackFPAD-TRAIN-IMC__TEST-O,
                                      Fed-StackFPAD-TRAIN-O__TEST-M)"
  echo "  --save_ckpt_freq            checkpoint saving frequency"
  echo "  --model                     model architecture (e.g., vit_base_patch16)"
  echo "  --split_type                split type (e.g., central, federated)"
  echo "  --n_clients                 number of federated clients"
  echo "  --max_communication_rounds  total communication rounds"
  echo "  --finetune                  path to finetune checkpoint file"
  echo "  --datapath                  path to dataset directory"
  echo "  --output_dir                base output directory for experiment results"
  echo "  --blr_min                   minimum base learning rate"
  echo "  --blr_max                   maximum base learning rate"
  echo "  --head_init_std_min         minimum head init std"
  echo "  --head_init_std_max         maximum head init std"
  echo "  --batch_sizes               comma-separated batch size options (e.g., 32,64,128)"
  echo "  --warmup_epochs             comma-separated warmup epochs (e.g., 10,20)"
  echo "  --layer_decays              comma-separated layer decay rates (e.g., 0.6,0.7,0.8)"
  echo "  --weight_decay              weight decay"
  echo "  --drop_path                 drop path rate"
  echo "  --reprob                    random erasing probability"
  echo "  --mixup                     mixup alpha"
  echo "  --cutmix                    cutmix alpha"
  echo "  --E_epoch                   number of local epochs"
  echo "  --num_local_clients         number of local clients (-1 for all)"
  echo "  --wandb_user                user for wandb platform"
  echo "  --wandb_key                 key for logging on wandb platform"
  echo ""
  exit 0
fi

# Sample usage:
# ./optuna_hyper_parameter_tuning.sh \
# --dataset Fed-StackFPAD-TRAIN-IOC__TEST-M \
# --save_ckpt_freq 50 \
# --model vit_base_patch16 \
# --split_type federated \
# --n_clients 3 \
# --max_communication_rounds 100 \
# --finetune /path/to/basemodel/base_vit_mae_imagenet22k_model.pth \
# --datapath /path/to/dataset \
# --output_dir /path/to/output \
# --blr_min 1e-5 \
# --blr_max 1e-2 \
# --head_init_std_min 1e-5 \
# --head_init_std_max 1e-2 \
# --batch_sizes "32,64,128" \
# --warmup_epochs "10,20" \
# --layer_decays "0.6,0.65,0.7" \
# --weight_decay 0.05 \
# --drop_path 0.1 \
# --reprob 0.25 \
# --mixup 0.8 \
# --cutmix 1.0 \
# --E_epoch 1 \
# --num_local_clients -1 \
# --wandb_user your_wandb_user_here \
# --wandb_key your_wandb_key_here

python ../hyper_parameter_optimization/optuna_hyper_parameter_sweep.py "$@"