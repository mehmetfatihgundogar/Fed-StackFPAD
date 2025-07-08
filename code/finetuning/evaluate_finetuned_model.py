# --------------------------------------------------------
# Based on SSL-FL (Self-Supervised Federated Learning) codebase
# Original Repository: https://github.com/rui-yan/SSL-FL
#
# This version maintained by: Mehmet Fatih Gündoğar
# Contributions:
# - Adapted ViT-MAE encoder for Face Presentation Attack Detection (FPAD) tasks
# - Integrated federated training and data center-specific fine-tuning logic
# - Developed stacking-based ensemble layer for robust inference
# - Added FPAD-specific preprocessing and evaluation pipeline
# --------------------------------------------------------

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.5.4"  # version check

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import finetuning.models_vit as models_vit
import util.misc as misc
from util.federated_averaging import Partial_Client_Selection, valid
from util.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
from util.start_config import print_options

import wandb


def get_args():
    parser = argparse.ArgumentParser('Fed-MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_name', default='mae', type=str)
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters\
    parser.add_argument('--data_set', default='replayattack', type=str,
                        help='FPAD dataset path')
    parser.add_argument('--data_path', default='/../../Datasets/replayattack', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # FL related parameters
    parser.add_argument("--n_clients", default=5, type=int, help="Number of clients")
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,
                        help="Total communication rounds.")
    parser.add_argument("--num_local_clients", default=-1, choices=[10, -1], type=int,
                        help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, default="central", help="Which data partitions to use")

    parser.add_argument("--predictions_dir", default="", help="Specify the directory for saving predictions data")
    parser.add_argument("--tsne_dir", default="", help="Specify the directory for saving data for tsne plot")
    parser.add_argument("--roc_dir", default="", help="Specify the directory for saving data for roc plot")
    parser.add_argument("--project", default="Fed-StackFPAD", help="Specify the project name")
    parser.add_argument("--wandb_key", default="", help="Specify the wandb key here")

    return parser.parse_args()


def main(args):
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.predictions_dir:
        Path(args.predictions_dir).mkdir(parents=True, exist_ok=True)
    if args.tsne_dir:
        Path(args.tsne_dir).mkdir(parents=True, exist_ok=True)
    if args.roc_dir:
        Path(args.roc_dir).mkdir(parents=True, exist_ok=True)

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}

    wand_run_name: str = 'Finetune'
    if args.eval is True:
        wand_run_name = 'Evaluation'
    # Wandb Setup
    try:
        wandb.login(key=args.wandb_key)
    except:
        print(f"W&B init failed, switching to dryrun mode.")
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        # Set the project where this run will be logged
        project=args.project,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=str(wand_run_name + " " + args.data_set),
        # Track hyperparameters and run metadata
        config={
            "base_learning_rate": args.blr,
            "learning_rate": args.lr,
            "architecture": args.model,
            "dataset": args.data_set,
            "epochs": args.max_communication_rounds,
            "min_lr": args.min_lr,
            "batch_size": args.batch_size,
            "split_type": args.split_type,
            "finetune": args.finetune,
            "eval": args.eval,
            "seed": args.seed,
        })

    # define our custom x axis metric
    wandb.define_metric("epoch")
    # define which metrics will be plotted against it
    wandb.define_metric("train*", step_metric="epoch")
    wandb.define_metric("validation_acc", step_metric="epoch")
    wandb.define_metric("validation_loss", step_metric="epoch")
    wandb.define_metric("eer", step_metric="epoch")
    wandb.define_metric("eer_threshold", step_metric="epoch")
    wandb.define_metric("hter", step_metric="epoch")
    wandb.define_metric("auc", step_metric="epoch")
    wandb.define_metric("balanced_acc", step_metric="epoch")
    wandb.define_metric("fpr", step_metric="epoch")
    wandb.define_metric("fnr", step_metric="epoch")
    wandb.define_metric("tpr", step_metric="epoch")
    wandb.define_metric("tnr", step_metric="epoch")
    wandb.define_metric("eer_mv", step_metric="epoch")
    wandb.define_metric("eer_threshold_mv", step_metric="epoch")
    wandb.define_metric("hter_mv", step_metric="epoch")
    wandb.define_metric("auc_mv", step_metric="epoch")
    wandb.define_metric("balanced_acc_mv", step_metric="epoch")
    wandb.define_metric("fpr_mv", step_metric="epoch")
    wandb.define_metric("fnr_mv", step_metric="epoch")
    wandb.define_metric("tpr_mv", step_metric="epoch")
    wandb.define_metric("tnr_mv", step_metric="epoch")

    misc.init_distributed_mode(args)

    # fix the seed for reproducibility
    misc.fix_random_seeds(args)

    cudnn.benchmark = True

    # prepare dataset
    create_dataset_and_evalmetrix(args, mode='finetune')

    dataset_val = DatasetFLFinetune(args=args, phase='validation')
    dataset_test = DatasetFLFinetune(args=args, phase='test')

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None

    # configuration for FedAVG, prepare model, optimizer, scheduler
    model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all = Partial_Client_Selection(args, model,
                                                                                                      mode='finetune')

    # ---------- Evaluation Phase
    print("=============== Running evaluation ===============")
    tot_clients = args.dis_cvs_files
    print('total_clients: ', tot_clients)
    epoch = -1

    while True:
        print('epoch: ', epoch)
        epoch += 1

        # ---- get dataset for each client for evaluation
        dataset_train = DatasetFLFinetune(args=args, phase='train')

        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        # ---- prepare model for a client
        model = list(model_all.values())[0]
        optimizer = list(optimizer_all.values())[0]
        loss_scaler = list(loss_scaler_all.values())[0]

        if args.distributed:
            model_without_ddp = model.module
            data_loader_train.sampler.set_epoch(epoch)
        else:
            model_without_ddp = model

        if args.eval:
            misc.load_model(args=args, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=None)

            validation_stats = valid(args, model, data_loader_val, epoch=epoch, save_prediction_data=True)

            test_stats = valid(args, model, data_loader_test,
                               validation_eer=validation_stats['validation_eer'],
                               validation_eer_threshold=validation_stats['validation_eer_threshold'],
                               epoch=epoch, save_prediction_data=True)
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")

            wandb.log({"test_acc": test_stats['acc1'], "loss": test_stats['loss']})

            model.cpu()

            exit(0)


if __name__ == '__main__':
    args = get_args()

    # run evaluation
    main(args)
