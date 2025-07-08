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

import os
import csv


def list_files_to_csv(directory, output_csv):
    file_list = []

    # Walk through the directory and collect filenames
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_list.append(file_path)

    # Write the filenames to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for file in file_list:
            writer.writerow([file])

    print(f'All filenames from {directory} have been written to {output_csv}')


def list_filenames_to_csv(directory, output_csv):
    file_list = []

    # Walk through the directory and collect filenames
    for root, _, files in os.walk(directory):
        for filename in files:
            file_list.append(filename)

    # Write the filenames to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for file in file_list:
            writer.writerow([file])

    print(f'All filenames from {directory} have been written to {output_csv}')


def generate_labels_csv(root_dir, output_csv):
    file_list = []

    # Walk through the directory and collect filenames
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.npy'):
                # Determine label based on whether 'attack' is in the filename
                label = 0.0 if 'attack' in filename.lower() else 1.0
                file_list.append((filename, label))

    # Write the filenames and labels to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for file, label in file_list:
            writer.writerow([file, label])

    print(f'Labels for all filenames in {root_dir} have been written to {output_csv}')
