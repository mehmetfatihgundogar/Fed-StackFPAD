# --------------------------------------------------------
# Fed-StackFPAD: Federated Learning for Face Presentation Attack Detection with Stacking to Tackle Data Heterogeneity
#
# Author: Mehmet Fatih Gündoğar
#
# Description:
# This file is part of the Fed-StackFPAD framework, a federated learning pipeline  designed for robust face presentation
# attack detection (FPAD). It combines self-supervised pretrained ViT encoders with local fine-tuning and a final
# stacking-based ensemble classifier.
#
# --------------------------------------------------------

import argparse
import optuna
import subprocess
from datetime import datetime

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--save_ckpt_freq", type=int, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--split_type", required=True)
parser.add_argument("--n_clients", type=int, required=True)
parser.add_argument("--max_communication_rounds", type=int, required=True)
parser.add_argument("--weight_decay", type=float, required=True)
parser.add_argument("--drop_path", type=float, required=True)
parser.add_argument("--reprob", type=float, required=True)
parser.add_argument("--mixup", type=float, required=True)
parser.add_argument("--cutmix", type=float, required=True)
parser.add_argument("--E_epoch", type=int, required=True)
parser.add_argument("--num_local_clients", type=int, required=True)
parser.add_argument("--optuna_trials", type=int, required=True)

# Paths
parser.add_argument("--finetune", required=True)
parser.add_argument("--datapath", required=True)
parser.add_argument("--output_dir", required=True)

# Hyperparameter search space
parser.add_argument("--blr_min", type=float, required=True)
parser.add_argument("--blr_max", type=float, required=True)
parser.add_argument("--head_init_std_min", type=float, required=True)
parser.add_argument("--head_init_std_max", type=float, required=True)

parser.add_argument("--batch_sizes", type=str, required=True)
parser.add_argument("--warmup_epochs", type=str, required=True)
parser.add_argument("--layer_decays", type=str, required=True)
parser.add_argument("--wandb_user", type=str, required=True)
parser.add_argument("--wandb_key", type=str, required=True)

args = parser.parse_args()

# Convert comma-separated lists
batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
warmup_epochs = [int(x.strip()) for x in args.warmup_epochs.split(",")]
layer_decays = [float(x.strip()) for x in args.layer_decays.split(",")]

current_time = datetime.now().strftime("%d%m%Y%H%M")
wandb_project_name = f"HypSweep-{args.dataset}-{current_time}"


def objective(trial):
    blr = trial.suggest_loguniform("blr", args.blr_min, args.blr_max)
    head_init_std = trial.suggest_loguniform("head_init_std", args.head_init_std_min, args.head_init_std_max)
    batch_size = trial.suggest_categorical("batch_size", batch_sizes)
    warmup_epoch = trial.suggest_categorical("warmup_epochs", warmup_epochs)
    layer_decay = trial.suggest_categorical("layer_decay", layer_decays)

    output_dir = f"{args.output_dir}/finetune_epoch{args.max_communication_rounds}_blr{blr}_bs{batch_size}_we{warmup_epoch}_headinit{head_init_std}"

    cmd = [
        "python", "../finetuning/run_federated_finetuning.py",
        "--finetune", args.finetune,
        "--layer_decay", str(layer_decay),
        "--drop_path", str(args.drop_path),
        "--reprob", str(args.reprob),
        "--mixup", str(args.mixup),
        "--cutmix", str(args.cutmix),
        "--data_path", args.datapath,
        "--data_set", args.dataset,
        "--output_dir", output_dir,
        "--blr", str(blr),
        "--batch_size", str(batch_size),
        "--save_ckpt_freq", str(args.save_ckpt_freq),
        "--max_communication_rounds", str(args.max_communication_rounds),
        "--split_type", args.split_type,
        "--model", args.model,
        "--warmup_epochs", str(warmup_epoch),
        "--weight_decay", str(args.weight_decay),
        "--n_clients", str(args.n_clients),
        "--E_epoch", str(args.E_epoch),
        "--num_local_clients", str(args.num_local_clients),
        "--head_init_std", str(head_init_std),
        "--project", wandb_project_name
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        eer = None
        run_id = None

        for line in output.split("\n"):
            if "Best EER:" in line:
                eer = float(line.split("Best EER:")[-1].strip())
            if "WandB Run ID:" in line:
                run_id = line.split("WandB Run ID:")[-1].strip()

        if eer is None or run_id is None:
            raise ValueError("EER or WandB Run ID not found!")

        wandb_url = f"https://wandb.ai/{args.wandb_user}/{wandb_project_name}/runs/{run_id}"
        print(f"WandB Link: {wandb_url}")
        trial.set_user_attr("wandb_url", wandb_url)

        return eer

    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)
        return 1.0


study = optuna.create_study(direction="minimize")
optuna.logging.set_verbosity(optuna.logging.DEBUG)
study.optimize(objective, n_trials=args.optuna_trials, catch=(Exception,))

best_wandb_link = study.best_trial.user_attrs["wandb_url"]
print("Best Hyperparameters:", study.best_params)
print(f"Best WandB Run Link: {best_wandb_link}")
