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

import torch

from torchvision import transforms

from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True


RETINA_MEAN = (0.5007, 0.5010, 0.5019)
RETINA_STD = (0.0342, 0.0535, 0.0484)

class DataAugmentationForPretrain(object):
    """ data transformations for pre-training"""
    def __init__(self, args):

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if args.model_name == 'mae':
            self.common_transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.RandomHorizontalFlip(p=0.5)])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        
        self.args = args
    
    def __call__(self, image):
        if self.args.model_name == 'mae':
            for_patches = self.common_transform(image)
            return self.patch_transform(for_patches)

    def __repr__(self):
        if self.args.model_name == 'mae':
            repr = "(DataAugmentationFoMAE,\n"
            repr += "  common_transform = %s,\n" % str(self.common_transform)
            repr += "  patch_transform = %s,\n" % str(self.patch_transform)

        return repr



def build_transform(is_train, mode, args):
    """ data transformations for fine-tuning"""

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    if mode == 'finetune':
        if is_train:
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.6, 1.)),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=args.input_size),
                transforms.CenterCrop(size=(args.input_size, args.input_size)), 
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
                ])

    return transform
