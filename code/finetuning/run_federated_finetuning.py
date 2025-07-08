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
import datetime
import json
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.5.4"  # version check
from copy import deepcopy

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import models_vit as models_vit
from engine_for_finetuning import train_one_epoch
import util.misc as misc
from util.federated_averaging import Partial_Client_Selection, valid, average_model
from util.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
from util.start_config import print_options

import wandb
from torch.distributed import get_rank


def get_args():
    parser = argparse.ArgumentParser('Federated fine-tuning for face presentation attack detection', add_help=False)
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
    parser.add_argument('--head_init_std', type=float, default=2e-5,
                        help='Std deviation for head (linear classifier) layer')

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

    try:
        rank = torch.cuda.current_device()
        print("GPU_ID:", rank)
    except:
        print("Unable to get rank info (perhaps non-distributed running). rank is set as 0")
        rank = 0

    if rank == 0:
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
            project=args.project,
            name=str(wand_run_name + " " + args.data_set),
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
                "warmup_epochs": args.warmup_epochs,
                "head_init_std": args.head_init_std,
                "layer_decay": args.layer_decay,
            },
            group=str(wand_run_name + " " + args.data_set),
        )

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
    else:
        os.environ['WANDB_MODE'] = 'disabled'

    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    misc.fix_random_seeds(args)

    cudnn.benchmark = True

    # prepare dataset
    create_dataset_and_evalmetrix(args, mode='finetune')

    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val = DatasetFLFinetune(args=args, phase='validation')

    if args.eval:
        dataset_test = DatasetFLFinetune(args=args, phase='test')
    else:
        dataset_test = None

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
    else:
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
    model_avg = deepcopy(model).cpu()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print("=============== Running fine-tuning ===============")
    tot_clients = args.dis_cvs_files
    print('total_clients: ', tot_clients)
    epoch = -1

    start_time = time.time()
    min_eer = 1.0
    max_acc = 0.0

    while True:
        print('epoch: ', epoch)
        epoch += 1

        if args.num_local_clients == len(args.dis_cvs_files):
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()

        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]

        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            print('cur_single_client: ', cur_single_client)
            print('proxy_single_client: ', proxy_single_client)

            args.single_client = cur_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens

            dataset_train = DatasetFLFinetune(args=args, phase='train')

            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()

            print(f'=========client: {proxy_single_client} ==============')
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
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            criterion = criterion_all[proxy_single_client]
            loss_scaler = loss_scaler_all[proxy_single_client]
            mixup_fn = mixup_fn_all[proxy_single_client]

            if args.distributed:
                model_without_ddp = model.module
            else:
                model_without_ddp = model

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            total_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
            num_training_steps_per_inner_epoch = len(dataset_train) // total_batch_size
            print("LR = %.8f" % args.lr)
            print("Batch size = %d" % total_batch_size)
            print("Number of training examples = %d" % len(dataset_train))
            print("Number of training training per epoch = %d" % num_training_steps_per_inner_epoch)

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch)

            if args.eval:
                misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler, model_ema=None)

                validation_stats = valid(args, model, data_loader_val, epoch=epoch, rank=rank)

                test_stats = valid(args, model, data_loader_test,
                                   validation_eer=validation_stats['validation_eer'],
                                   validation_eer_threshold=validation_stats['validation_eer_threshold'],
                                   epoch=epoch, rank=rank)

                model.cpu()

                exit(0)

            for inner_epoch in range(args.E_epoch):
                # ============ training one epoch  ============
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args.clip_grad, proxy_single_client,
                    mixup_fn,
                    log_writer=log_writer,
                    args=args
                )

                # ============ writing logs ============
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'client': cur_single_client,
                             'epoch': epoch,
                             'inner_epoch': inner_epoch,
                             'n_parameters': n_parameters}

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

        # =========== model average and eval ============
        # average model
        average_model(args, model_avg, model_all)

        # save the global model
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.max_communication_rounds:
                misc.save_model(
                    args=args, model=model_avg, model_without_ddp=model_avg,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        if data_loader_val is not None:
            model_avg.to(args.device)
            validation_stats = valid(args, model_avg, data_loader_val, epoch=epoch, rank=rank, min_eer=min_eer)

            if min_eer > validation_stats['validation_eer']:
                min_eer = validation_stats['validation_eer']
                if args.output_dir:
                    print("Best model is changed due to the min_eer and it is saved on epoch:", epoch)
                    misc.save_model(
                        args=args, model=model_avg,
                        model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=None)
            elif min_eer == validation_stats['validation_eer'] and max_acc < validation_stats['acc1']:
                max_acc = validation_stats['acc1']
                print("Best model is changed due to the same min_eer and improved accuracy and it is saved on epoch:",
                      epoch)
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model_avg,
                        model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=None)

            print(f'Min eer: {min_eer:.2f}%')
            if log_writer is not None:
                log_writer.update(test_acc1=validation_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=validation_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=validation_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'test_{k}': v for k, v in validation_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        model_avg.to('cpu')

        print('global_step_per_client: ', args.global_step_per_client[proxy_single_client])
        print('t_total: ', args.t_total[proxy_single_client])

        if args.global_step_per_client[proxy_single_client] >= args.t_total[proxy_single_client]:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            if rank == 0:
                run_id = wandb.run.id
                wandb.finish()

            print(f"Best EER: {min_eer}")
            print(f"WandB Run ID: {run_id}")
            return min_eer


if __name__ == '__main__':
    args = get_args()

    main(args)
