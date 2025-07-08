#!/bin/bash

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./federated_finetune.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --data_path                 path to dataset (e.g.,
                                      /path/to/datasets/fed-stackfpad__exp_mode_singleclient_train_on_O__test_on_M,
                                      /path/to/datasets/fed-stackfpad__exp_mode_multiclient_train_on_IOC__test_on_M,
                                      /path/to/datasets/fed-stackfpad__exp_mode_centralized_train_on_OCI__test_on_M)"
  echo "  --data_set                  dataset name for the experiment (e.g.,
                                      Fed-StackFPAD-TRAIN-IMC__TEST-O,
                                      Fed-StackFPAD-TRAIN-O__TEST-M)"
  echo "  --resume                    path to finetuned model checkpoint"
  echo "  --nb_classes                number of classes (e.g., 2)"
  echo "  --output_dir                output directory"
  echo "  --save_ckpt_freq            checkpoint saving frequency"
  echo "  --model                     model architecture (e.g., vit_base_patch16)"
  echo "  --batch_size                batch size"
  echo "  --split_type                split type (e.g., central, federated)"
  echo "  --blr                       base learning rate"
  echo "  --head_init_std             std deviation for head (linear classifier) layer"
  echo "  --warmup_epochs             epochs to warmup LR"
  echo "  --layer_decay               layer decay rate"
  echo "  --weight_decay              weight decay"
  echo "  --drop_path                 drop path rate"
  echo "  --reprob                    random erasing probability"
  echo "  --mixup                     mixup alpha"
  echo "  --cutmix                    cutmix alpha"
  echo "  --n_clients                 number of federated clients"
  echo "  --E_epoch                   number of local epochs"
  echo "  --max_communication_rounds  maximum communication rounds"
  echo "  --num_local_clients         number of local clients (-1 for all)"
  echo "  --project                   project name for wandb setup"
  echo "  --wandb_key                 key for logging on wandb platform"
  exit 0
fi

# Sample usage:
# ./federated_finetune.sh \
# --data_path /path/to/datasets/fed-stackfpad__exp_mode_multiclient_train_on_IOC__test_on_M \
# --data_set Fed-StackFPAD-TRAIN-IOC__TEST-M \
# --finetune /path/to/basemodel/base_vit_mae_imagenet22k_model.pth \
# --nb_classes 2 \
# --output_dir /path/to/basemodel \
# --save_ckpt_freq 2 \
# --model vit_base_patch16 \
# --batch_size 64 \
# --split_type central \
# --blr 3e-3 \
# --head_init_std 2e-3 \
# --warmup_epochs 20 \
# --layer_decay 0.65 \
# --weight_decay 0.05 \
# --drop_path 0.1 \
# --reprob 0.25 \
# --mixup 0.8 \
# --cutmix 1.0 \
# --n_clients 3 \
# --E_epoch 1 \
# --max_communication_rounds 100 \
# --num_local_clients -1 \
# --project fed_stackfpad \
# --wandb_key your_wandb_key_here

python ../finetuning/run_federated_finetuning.py "$@"