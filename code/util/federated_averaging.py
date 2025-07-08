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

from __future__ import absolute_import, division, print_function
import os
import numpy as np
from copy import deepcopy
import torch
import wandb

from .lars import LARS
from . import misc as misc
from .lr_decay import param_groups_lrd
from .misc import NativeScalerWithGradNormCount as NativeScaler
from .pos_embed import interpolate_pos_embed
from .optim_factory import add_weight_decay

from timm.utils import accuracy
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_

from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from dataset_preprocessing.datasets import prediction_index_map

import torch.nn.functional as F

import matplotlib.pyplot as plt


def Partial_Client_Selection(args, model, mode='pretrain'):
    device = torch.device(args.device)

    # Select partial clients join in FL train
    if args.num_local_clients == -1:  # all the clients joined in the train
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients = len(args.dis_cvs_files)  # update the true number of clients
    else:
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]

    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    criterion_all = {}
    lr_scheduler_all = {}
    wd_scheduler_all = {}
    loss_scaler_all = {}
    mixup_fn_all = {}
    args.learning_rate_record = {}
    args.t_total = {}

    # Load pretrained model if mode='finetune'
    if (mode == 'finetune' or mode == 'linprob') and args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if args.model_name == 'mae':
            checkpoint_model = checkpoint['model']

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        if args.model_name == 'mae':
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        if mode == 'finetune':
            #trunc_normal_(model.head.weight, std=2e-5)

            # Freeze All Blocks Except Linear Classifier
            # ------------------------------------------
            """
            for param in model.parameters():
                param.requires_grad = False

            for param in model.head.parameters():
                param.requires_grad = True
            # ------------------------------------------
            """
            trunc_normal_(model.head.weight, std=args.head_init_std)
        elif mode == 'linprob':
            trunc_normal_(model.head.weight, std=0.01)

            # for linear prob only
            # hack: revise model's head with BN
            model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
                                             model.head)
            # freeze all but the head
            for _, p in model.named_parameters():
                p.requires_grad = False
            for _, p in model.head.named_parameters():
                p.requires_grad = True

    if args.distributed:
        if args.sync_bn:  #activate synchronized batch norm
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    for proxy_single_client in args.proxy_clients:

        global_rank = misc.get_rank()
        num_tasks = misc.get_world_size()

        print('clients_with_len: ', args.clients_with_len[proxy_single_client])

        if args.model_name == 'mae':
            total_batch_size = args.batch_size * args.accum_iter * num_tasks
            if args.lr is None:  # only base_lr is specified
                args.lr = args.blr * total_batch_size / 256

        num_training_steps_per_inner_epoch = args.clients_with_len[proxy_single_client] // total_batch_size

        print("Batch size = %d" % total_batch_size)
        print("Number of training steps = %d" % num_training_steps_per_inner_epoch)
        print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_inner_epoch))

        # model_all
        model_all[proxy_single_client] = deepcopy(model)
        model_all[proxy_single_client] = model_all[proxy_single_client].to(device)

        if args.distributed:
            model_all[proxy_single_client] = torch.nn.parallel.DistributedDataParallel(model_all[proxy_single_client],
                                                                                       device_ids=[args.gpu],
                                                                                       find_unused_parameters=True)

        if args.distributed:
            model_without_ddp = model_all[proxy_single_client].module
        else:
            model_without_ddp = model_all[proxy_single_client]

        # optimizer_all
        if mode == 'pretrain':
            if args.model_name == 'mae':
                param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
                optimizer_all[proxy_single_client] = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

        elif mode == 'finetune':
            if args.model_name == 'mae':
                # build optimizer with layer-wise lr decay (lrd)
                param_groups = param_groups_lrd(model_without_ddp, args.weight_decay,
                                                no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                                layer_decay=args.layer_decay
                                                )
                optimizer_all[proxy_single_client] = torch.optim.AdamW(param_groups, lr=args.lr)
        elif mode == 'linprob':
            if args.model_name == 'mae':
                optimizer_all[proxy_single_client] = LARS(model_without_ddp.head.parameters(), lr=args.lr,
                                                          weight_decay=args.weight_decay)

        if mode == 'finetune':
            mixup_fn = None
            mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
            if mixup_active:
                print("Mixup is activated!")
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                    prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                    label_smoothing=args.smoothing, num_classes=args.nb_classes)
            mixup_fn_all[proxy_single_client] = mixup_fn

            if mixup_fn is not None:
                # smoothing is handled with mixup label transform
                criterion = SoftTargetCrossEntropy()
            elif args.smoothing > 0.:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()
            criterion_all[proxy_single_client] = criterion

        if mode == 'linprob':
            criterion_all[proxy_single_client] = torch.nn.CrossEntropyLoss()

        # loss_scaler_all
        loss_scaler_all[proxy_single_client] = NativeScaler()

        # ---- resume from an existing checkpoint
        if mode == 'pretrain' and args.resume:
            print("Resuming from an existing checkpoint...")
            misc.load_model(args=args, model_without_ddp=model_all[proxy_single_client],
                            optimizer=optimizer_all[proxy_single_client],
                            loss_scaler=loss_scaler_all[proxy_single_client], model_ema=None)

        # get the total decay steps first
        args.t_total[
            proxy_single_client] = num_training_steps_per_inner_epoch * args.E_epoch * args.max_communication_rounds

        args.learning_rate_record[proxy_single_client] = []

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    if args.model_name == 'mae':
        if mode == 'pretrain':
            return model_all, optimizer_all, loss_scaler_all
        else:
            return model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all


def average_model(args, model_avg, model_all):
    model_avg.cpu()
    print('Calculate the model avg----')
    params = dict(model_avg.named_parameters())

    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
            single_client = args.proxy_clients[client]

            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()

            if client == 0:
                if args.distributed:
                    tmp_param_data = dict(model_all[single_client].module.named_parameters())[
                                         name].data * single_client_weight
                else:
                    tmp_param_data = dict(model_all[single_client].named_parameters())[
                                         name].data * single_client_weight
            else:
                if args.distributed:
                    tmp_param_data = tmp_param_data + \
                                     dict(model_all[single_client].module.named_parameters())[
                                         name].data * single_client_weight
                else:
                    tmp_param_data = tmp_param_data + \
                                     dict(model_all[single_client].named_parameters())[
                                         name].data * single_client_weight

        params[name].data.copy_(tmp_param_data)

    print('Update each client model parameters----')

    for single_client in args.proxy_clients:

        if args.distributed:
            tmp_params = dict(model_all[single_client].module.named_parameters())
        else:
            tmp_params = dict(model_all[single_client].named_parameters())

        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name))

    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)


def valid(args, model, data_loader, validation_eer=None, validation_eer_threshold=None, epoch=None,
          save_prediction_data=False, rank=None, min_eer=1.0):
    # eval_losses = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'
    if validation_eer is not None:
        header = 'Test:'
    # switch to evaluation mode
    model.eval()

    print("++++++ Running Validation ++++++")

    scores = []
    true_labels = []
    image_names = []  # To store image names for majority voting
    features = [] # Encoder features for T-SNE plot

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        names = batch[2]

        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)
            if save_prediction_data:
                feature_vec = model.forward_features(images).cpu()  # shape: [B, 768]
                features.append(feature_vec)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        positive_scores = output[:, 1]
        scores.extend(positive_scores.cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        image_names.extend(names)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)


    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    final_preds = np.array(scores)
    final_trues = np.array(true_labels)

    train_set = args.data_set.split("TRAIN-")[1].split("__TEST-")[0]
    test_set = args.data_set.split("__TEST-")[1]
    if save_prediction_data:
        key = (train_set, test_set)
        index = prediction_index_map[key]
        # Save each prediction set individually
        import pickle
        if validation_eer_threshold is not None:
            with open(f"{args.predictions_dir}/prediction_data_test_target_{test_set}_{index}.pkl", "wb") as file:
                pickle.dump({"names": image_names, "pred": final_preds, "true_labels": final_trues}, file)
        else:
            with open(f"{args.predictions_dir}/prediction_data_validation_target_{test_set}_{index}.pkl", "wb") as file:
                pickle.dump({"names": image_names, "pred": final_preds, "true_labels": final_trues}, file)

    if validation_eer_threshold is not None:
        if save_prediction_data:
            # save data for t-sne plot
            model_name = f'{args.split_type}_{train_set}_{test_set}'
            features = torch.cat(features, dim=0).numpy()
            np.save(f"{args.tsne_dir}/features_{model_name}.npy", features)
            np.save(f"{args.tsne_dir}/labels_{model_name}.npy", final_trues)
            np.save(f"{args.tsne_dir}/ids_{model_name}.npy", np.array(image_names, dtype=str))


        # Calculate predictions based on the given EER threshold
        predictions = (final_preds > validation_eer_threshold).astype(int)

        # Calculate FPR and FNR based on the given threshold
        fpr = np.sum((predictions == 1) & (final_trues == 0)) / np.sum(final_trues == 0)
        fnr = np.sum((predictions == 0) & (final_trues == 1)) / np.sum(final_trues == 1)

        # Calculate HTER based on the FPR and FNR
        hter_with_threshold = (fpr + fnr) / 2

        # AUC calculation before majority voting
        auc = roc_auc_score(final_trues, final_preds)

        # Balanced Accuracy calculation before majority voting
        balanced_acc = balanced_accuracy_score(final_trues, (final_preds > validation_eer_threshold).astype(int))

        if rank == 0:
            wandb.log({"hter_with_threshold": hter_with_threshold, "auc": auc,
                       "validation_eer": validation_eer, "balanced_acc": balanced_acc,
                       "fpr": fpr, "fnr": fnr, "tpr": 1 - fnr, "tnr": 1 - fpr, "epoch": epoch, "min_eer": validation_eer})

        print("hter_with_threshold:", hter_with_threshold, ", auc:", auc,
              " validation_eer:", validation_eer, " balanced_acc:", balanced_acc,
              " fpr:", fpr, " fnr:", fnr, " tpr:", 1 - fnr, " tnr:", 1 - fpr)

        if save_prediction_data:
            # ROC Curve Calculation and Plotting with FPR on x-axis and FNR on y-axis
            model_name = f'{args.split_type}_{train_set}_{test_set}'
            fpr_values, tpr_values, _ = roc_curve(final_trues, final_preds)
            fnr_values = 1 - tpr_values  # FNR is the complement of TPR
            np.savez(f'{args.roc_dir}/{model_name}.npz', fpr=fpr_values, fnr=fnr_values)

    else:
        # EER and HTER calculation before majority voting
        fpr, tpr, thresholds = roc_curve(final_trues, final_preds)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        hter = (fpr[np.nanargmin(np.absolute(fnr - fpr))] + fnr[
            np.nanargmin(np.absolute(fnr - fpr))]) / 2

        # Log eer and eer threshold for hter calculation on test fold.
        metric_logger.add_meter('validation_eer', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('validation_eer_threshold', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.update(validation_eer=float(eer))
        metric_logger.update(validation_eer_threshold=float(eer_threshold))

        # AUC calculation before majority voting
        auc = roc_auc_score(final_trues, final_preds)

        # Balanced Accuracy calculation before majority voting
        balanced_acc = balanced_accuracy_score(final_trues, (final_preds > eer_threshold).astype(int))

        if rank == 0:
            wandb.log({"validation_acc": metric_logger.acc1.global_avg, "validation_loss": metric_logger.loss.median,
                       "eer": eer, "eer_threshold": eer_threshold, "hter": hter, "auc": auc,
                       "balanced_acc": balanced_acc, "fpr": fpr[np.nanargmin(np.absolute(fnr - fpr))],
                       "fnr": fnr[np.nanargmin(np.absolute(fnr - fpr))], "tpr": 1 - fnr[np.nanargmin(np.absolute(fnr - fpr))],
                       "tnr": 1 - fpr[np.nanargmin(np.absolute(fnr - fpr))],
                       "epoch": epoch, "min_eer": min_eer})
        print("validation_acc:", metric_logger.acc1.global_avg, ", validation_loss:", metric_logger.loss.median,
              " eer:", eer, " eer_threshold:",
              eer_threshold, " hter:", hter, f"auc: {auc:.4f}", " balanced_acc:", balanced_acc,
              " fpr:", fpr[np.nanargmin(np.absolute(fnr - fpr))], " fnr:", fnr[np.nanargmin(np.absolute(fnr - fpr))],
              " tpr:", 1 - fnr[np.nanargmin(np.absolute(fnr - fpr))], " tnr:",
              1 - fpr[np.nanargmin(np.absolute(fnr - fpr))])


    # Majority voting based metric calculations
    frame_results = defaultdict(lambda: {'pred': [], 'true': []})
    for name, score, true_label in zip(image_names, scores, true_labels):
        frame_id = name.rsplit('_', 1)[0]
        frame_results[frame_id]['pred'].append(score)
        frame_results[frame_id]['true'].append(true_label)

    final_preds_mv = []
    final_trues_mv = []

    for frame_id, result in frame_results.items():
        avg_pred = np.mean(result['pred'])
        majority_true = 1 if np.mean(result['true']) > 0.5 else 0
        final_preds_mv.append(avg_pred)
        final_trues_mv.append(majority_true)

    final_preds_mv = np.array(final_preds_mv)
    final_trues_mv = np.array(final_trues_mv)

    if validation_eer_threshold is not None:
        # Calculate predictions based on the given EER threshold
        predictions_mv = (final_preds_mv > validation_eer_threshold).astype(int)

        # Calculate FPR and FNR based on the given threshold
        fpr_mv = np.sum((predictions_mv == 1) & (final_trues_mv == 0)) / np.sum(final_trues_mv == 0)
        fnr_mv = np.sum((predictions_mv == 0) & (final_trues_mv == 1)) / np.sum(final_trues_mv == 1)

        # Calculate HTER based on the FPR and FNR
        hter_with_threshold_mv = (fpr_mv + fnr_mv) / 2

        # AUC calculation before majority voting
        auc_mv = roc_auc_score(final_trues_mv, final_preds_mv)

        # Balanced Accuracy calculation before majority voting
        balanced_acc_mv = balanced_accuracy_score(final_trues_mv, (final_preds_mv > validation_eer_threshold).astype(int))

        if rank == 0:
            wandb.log({"hter_with_threshold_mv": hter_with_threshold_mv, "auc_mv": auc_mv,
                       "validation_eer": validation_eer, "balanced_acc_mv": balanced_acc_mv,
                       "fpr_mv": fpr_mv, "fnr_mv": fnr_mv, "tpr_mv": 1 - fnr_mv, "tnr_mv": 1 - fpr_mv, "epoch": epoch})

        print("hter_with_threshold_mv:", hter_with_threshold_mv, ", auc:", auc_mv,
              " validation_eer:", validation_eer, " balanced_acc_mv:", balanced_acc_mv,
              " fpr_mv:", fpr_mv, " fnr_mv:", fnr_mv, " tpr_mv:", 1 - fnr_mv, " tnr_mv:", 1 - fpr_mv)
    else:
        # EER and HTER calculation after majority voting
        fpr_mv, tpr_mv, thresholds_mv = roc_curve(final_trues_mv, final_preds_mv)
        fnr_mv = 1 - tpr_mv
        eer_threshold_mv = thresholds_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))]
        eer_mv = fpr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))]
        hter_mv = (fpr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))] + fnr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))]) / 2

        # AUC calculation after majority voting
        auc_mv = roc_auc_score(final_trues_mv, final_preds_mv)

        # Balanced Accuracy calculation after majority voting
        balanced_acc_mv = balanced_accuracy_score(final_trues_mv, (final_preds_mv > eer_threshold_mv).astype(int))

        if rank == 0:
            wandb.log({"validation_acc": metric_logger.acc1.global_avg, "validation_loss": metric_logger.loss.median,
                       "eer_mv": eer_mv, "eer_threshold_mv": eer_threshold_mv, "hter_mv": hter_mv, "auc_mv": auc_mv,
                       "balanced_acc_mv": balanced_acc_mv, "fpr_mv": fpr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))],
                       "fnr_mv": fnr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))],
                       "tpr_mv": 1 - fnr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))],
                       "tnr_mv": 1 - fpr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))],
                       "epoch": epoch})
        print("validation_acc:", metric_logger.acc1.global_avg, ", validation_loss:", metric_logger.loss.median,
              " eer_mv:", eer_mv, " eer_threshold_mv:",  eer_threshold_mv,
              " hter_mv:", hter_mv, f"auc_mv: {auc_mv:.4f}", " balanced_acc_mv:", balanced_acc_mv,
              " fpr_mv:", fpr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))],
              " fnr_mv:", fnr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))],
              " tpr_mv:", 1 - fnr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))],
              " tnr_mv:", 1 - fpr_mv[np.nanargmin(np.absolute(fnr_mv - fpr_mv))])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
