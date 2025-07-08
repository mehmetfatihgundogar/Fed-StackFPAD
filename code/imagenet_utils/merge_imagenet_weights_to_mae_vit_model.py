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

import torch
import argparse
import os.path


def main(args):
    # Paths to the ImageNet and ViT .pth files
    imagenet_path = os.path.join(args.root_folder, args.imagenet_file)
    base_vit_model_path = os.path.join(args.root_folder, args.base_vit_model_file)
    output_path = os.path.join(args.root_folder, args.output_file)

    # Load the checkpoints
    imagenet_checkpoint = torch.load(imagenet_path, map_location="cpu")
    base_vit_model_checkpoint = torch.load(base_vit_model_path, map_location="cpu")

    # Extract state_dicts
    imagenet_state_dict = imagenet_checkpoint.get('model', imagenet_checkpoint)
    vit_state_dict = base_vit_model_checkpoint.get('model', base_vit_model_checkpoint)

    # Replace encoder weights in the ViT model with ImageNet weights
    for key, weight in imagenet_state_dict.items():
        if key in vit_state_dict and weight.shape == vit_state_dict[key].shape:
            print(f"Replacing {key}")
            vit_state_dict[key] = weight

    # Save the updated checkpoint
    base_vit_model_checkpoint['model'] = vit_state_dict
    torch.save(base_vit_model_checkpoint, output_path)
    print(f"Updated model saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ImageNet weights into MAE ViT model checkpoint.")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing the model files. Example: /path/to/models")
    parser.add_argument("--imagenet_file", type=str, required=True, help="File name of the MAE model pretrained on ImageNet-22k. Example: mae_pretrain_vit_base.pth")
    parser.add_argument("--base_vit_model_file", type=str, required=True, help="File name of the base ViT model. Example: base_vit_model.pth")
    parser.add_argument("--output_file", type=str, required=True, help="File name of the output model (ViT model with ImageNet-22k encoder weights). Example: vit_model_with_imagenet22k_encoder_weights.pth")
    args = parser.parse_args()
    main(args)
