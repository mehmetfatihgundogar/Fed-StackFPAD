# Fed-StackFPAD

**Federated Learning for Face Presentation Attack Detection with Stacking to Tackle Data Heterogeneity**  

IEEE Access, 2025. [HTML (upon acceptance)]() | [Cite](#citation)

**TL;DR:** PyTorch implementation of the Fed-StackFPAD framework for cross-domain face presentation attack detection (FPAD), integrating federated learning, self-supervised ViT pretraining on ImageNet-22K, and a stacking-based ensemble strategy to address domain shift and data heterogeneity.

- Our approach is the first to adapt a self-supervised pretrained ViT-MAE within a federated learning setting for FPAD.
- Leveraging strong generalization from large-scale pretraining, it effectively mitigates data heterogeneity without requiring additional pretraining on FPAD datasets.
- We introduce a stacking-based ensemble strategy that combines predictions from data center-specific models and the global federated model to address domain shift and improve decision boundaries.
- Extensive experiments across multiple FPAD benchmarks show that our method significantly reduces HTER compared to prior state-of-the-art approaches, demonstrating superior robustness in non-IID scenarios.

<img src="docs/fed-stackfpad.png" width="1024px" align="center" />

---

## Pre-requisites

### Environment Setup
Clone the repository and create the environment:

```bash
git clone https://github.com/your-username/Fed-StackFPAD.git
cd Fed-StackFPAD
conda env create -f environment.yml
conda activate fed-stackfpad
```

* NVIDIA GPU (Tested on NVIDIA Tesla V100 16GB GPUs via TRUBA HPC resources and NVIDIA RTX 6000 Ada Generation GPU on a local workstation)
* Python (3.8.12), torch (2.2.2), timm (0.5.4), numpy (1.21.2), pandas (1.4.2), scikit-learn (1.0.2), scipy (1.7.1), seaborn (0.11.2)

### Base ViT-B/16 and Fine-tuned ViT-B/16 Models
You can access and reproduce the results presented in the paper using the models available in  [Fed-StackFPAD/data](https://github.com/mehmetfatihgundogar/Fed-StackFPAD/blob/master/data/README.md).

### Training and Evaluation
You can find the necessary scripts for end-to-end training and evaluation in [Fed-StackFPAD/code/script](https://github.com/mehmetfatihgundogar/Fed-StackFPAD/tree/code/script). In the sections below, example script calls are provided assuming Replay-Attack, Oulu-Npu, and Casia-Fasd are configured as federated data centers and Msu-Mfsd is used as the test set. For different configurations, script parameters can be updated according to the options specified in each script’s help description.

#### Preprocessing of Datasets for an Experiment
Our code support below datasets:
1. Replay-Attack (I)
2. Msu-Mfsd (M)
3. Oulu-Npu (O)
4. Casia-Fasd (C)

Use the provided script to preprocess the datasets:
```bash
# For example, if you want to configure the Replay-Attack, Oulu-Npu, and Casia-Fasd datasets as federated data centers and designate Msu-Mfsd as the test set, the script below will prepare the data accordingly.

./prepare_datasets.sh --base_folder /path/to/datasets --num_of_frame_steps 5 --training_datasets 1,3,4 --test_dataset 2
```

#### Initialization of Encoder Weights of ViT-B/16 ViT Model with MAE Weights Pretrained on Imagenet-22k
* Download ViT-B/16 ViT Model:
* ``` [download](drive.google.com) ```
* Download MAE weights pretrained on ImageNet-22k:
* ``` wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth ```

Run the ViT model weight initialization script to initialize the encoder weights of the ViT-B/16 model:
```bash
/prepare_vit_model_based_on_imagenet22K_encoder_weights.sh --root_folder /path/to/models --imagenet_file mae_pretrain_vit_base.pth --base_vit_model_file base_vit_model.pth --output_file vit_model_with_imagenet22k_encoder_weights.pth
```

#### Federated Fine-Tuning
Use the provided script to perform federated fine-tuning:
```bash
./federated_finetune.sh \
--data_path /path/to/datasets/fed-stackfpad__exp_mode_multiclient_train_on_IOC__test_on_M \
--data_set Fed-StackFPAD-TRAIN-IOC__TEST-M \
--finetune /path/to/basemodel/base_vit_mae_imagenet22k_model.pth \
--nb_classes 2 \
--output_dir /path/to/basemodel \
--save_ckpt_freq 2 \
--model vit_base_patch16 \
--batch_size 64 \
--split_type central \
--blr 3e-3 \
--head_init_std 2e-3 \
--warmup_epochs 20 \
--layer_decay 0.65 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--n_clients 3 \
--E_epoch 1 \
--max_communication_rounds 100 \
--num_local_clients -1 \
--project fed_stackfpad \
--wandb_key your_wandb_key_here
```

#### Optuna Hyper-Parameter Optimization
Use the provided script to run Optuna hyperparameter optimization:
```bash
./optuna_hyper_parameter_tuning.sh \
--dataset Fed-StackFPAD-TRAIN-IOC__TEST-M \
--save_ckpt_freq 50 \
--model vit_base_patch16 \
--split_type federated \
--n_clients 3 \
--max_communication_rounds 100 \
--finetune /path/to/basemodel/base_vit_mae_imagenet22k_model.pth \
--datapath /path/to/dataset \
--output_dir /path/to/output \
--blr_min 1e-5 \
--blr_max 1e-2 \
--head_init_std_min 1e-5 \
--head_init_std_max 1e-2 \
--batch_sizes "8,16,32,64,128" \
--warmup_epochs "10,20" \
--layer_decays "0.6,0.65,0.7" \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--E_epoch 1 \
--num_local_clients -1 \
--wandb_user your_wandb_user_here \
--wandb_key your_wandb_key_here
```


#### Inference on Fine-tuned Models
Use the provided script to evaluate the fine-tuned models:
```bash
./evaluation_of_finetuned_model.sh \
--data_path /path/to/datasets/fed-stackfpad__exp_mode_multiclient_train_on_IOC__test_on_M \
--data_set Fed-StackFPAD-TRAIN-IOC__TEST-M \
--resume /path/to/finetunedmodel/checkpoint-best.pth \
--output_dir /path/to/finetunedmodel/eval_results \
--model vit_base_patch16 \
--batch_size 64 \
--split_type central \
--n_clients 3 \
--eval \
--predictions_dir /path/to/finetunedmodel/predictions \
--tsne_dir /path/to/finetunedmodel/tsne \
--roc_dir /path/to/finetunedmodel/roc_data \
--project fed_stackfpad \
--wandb_key your_wandb_key_here
```


#### Meta-Learning with Stacking Based Ensemble
Concat prediction files of fine-tuned models for stacking:
```bash
./concat_prediction_data_for_stacking.sh --target M --num_files 4 --prediction_data_folder /path/to/predictions

```

Use the provided script to perform stacked model training and evaluation:
```bash
./stacking.sh --target M --stacked_model_type Svm --train_file /path/to/predictions/validation.pkl --test_file /path/to/predictions/test.pkl --output_roc_file /path/to/roc/data/stacked_M.npz --output_model_file ./svm_model_M.pkl
```

#### Generation of t-SNE Plots
Use the provided script to generate t-SNE plot for all domains:
```bash
./generate_tsne_for_attack_vs_genuine.sh --feature_dir /path/to/tsne/data --split_type federated --train_set IOC --output_file tsne_attack_vs_genuine.pdf
```

Use the provided script to generate t-SNE plot for attack vs. genuine classes:
```bash
./generate_tsne_for_all_domains.sh --target_domain M --feature_dir /path/to/tsne/data --split_type federated --train_set IOC --output_file tsne_federated_IOC_M.pdf
```

#### Generation of ROC Plots
Use the provided scripts to generate ROC plots:
```bash
./generate_roc_plot.sh --input_dir /path/to/roc_data --target M --output_dir ./plots
```


## Acknowledgements
We sincerely thank the authors of:
* [SSL-FL](https://github.com/rui-yan/SSL-FL)

This project reuses parts of their open-source code with appropriate modifications.

## Citation
If you find our work helpful in your research or if you use any source codes or datasets, please cite our paper.

```bibtex
@article{gundogar2025fedstackfpad,
  title={Fed-StackFPAD: Federated Learning for Face Presentation Attack Detection with Stacking to Tackle Data Heterogeneity},
  author={Gündoğar, Mehmet Fatih and Eroğlu Erdem, Çiğdem and Korçak, Ömer},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```