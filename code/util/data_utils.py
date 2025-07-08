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

import numpy as np
import pandas as pd

import os

from .datasets import build_transform

from PIL import Image
from skimage.transform import resize
import cv2
import torch.utils.data as data


class DatasetFLFinetune(data.Dataset):
    """ data loader for fine-tuning """
    def __init__(self, args, phase, mode='finetune'):
        super(DatasetFLFinetune, self).__init__()
        self.phase = phase
        is_train = (phase == 'train')

        if not is_train:
            args.single_client = os.path.join(args.data_path, f'{self.phase}.csv')

        if args.split_type == 'central':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients',
                                          args.split_type, args.single_client)
        
        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
        
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                        open(os.path.join(args.data_path, 'labels.csv'))}
        
        self.transform = build_transform(is_train, mode, args)
        
        self.args = args
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join(self.args.data_path, self.phase, self.img_paths[index])
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)
        
        img = np.load(path)
        
        if img.ndim < 3:
            img = np.concatenate((img,)*3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:,:,:3]
        
        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)
        return sample, target, name

    def __len__(self):
        return len(self.img_paths)


def create_dataset_and_evalmetrix(args, mode='finetune'):
    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = ['central']

    if args.split_type == 'central':
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
    else:
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type))
    
    args.clients_with_len = {}
    
    for single_client in args.dis_cvs_files:
        if args.split_type == 'central':
            img_paths = list({line.strip().split(',')[0] for line in
                            open(os.path.join(args.data_path, args.split_type, single_client))})
        else:
            img_paths = list({line.strip().split(',')[0] for line in
                                open(os.path.join(args.data_path, f'{args.n_clients}_clients',
                                                args.split_type, single_client))})
        args.clients_with_len[single_client] = len(img_paths)
    
    
    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}
    
    for single_client in args.dis_cvs_files:
        if mode == 'finetune':
            args.best_acc[single_client] = 0 if args.nb_classes > 1 else 999
            args.current_acc[single_client] = 0
            args.current_test_acc[single_client] = []
            args.best_eval_loss[single_client] = 9999


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_w:offset_w + size, offset_h:offset_h + size]