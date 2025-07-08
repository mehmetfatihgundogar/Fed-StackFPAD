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

import random
from datasets import Datasets, DatasetMap
import os
import cv2
import numpy as np
from PIL import Image


def extract_and_save_frames(source_dirs, target_dir, dataset=None, mode='step', step=None, resize_shape=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    video_path = os.path.join(root, file)
                    video_name = os.path.splitext(file)[0]
                    cap = cv2.VideoCapture(video_path)

                    # Get total number of frames
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    if mode == 'step' and step is not None:
                        # Extract frames at the given step interval
                        frame_indices = list(range(0, total_frames, step))
                    elif mode == 'middle':
                        # Extract only the middle frame
                        frame_indices = [total_frames // 2]
                    else:
                        # If mode or step is not defined, extract all frames
                        frame_indices = range(total_frames)

                    if dataset == Datasets.CasiaFasd.value:
                        dir_name = os.path.dirname(video_path)
                        client_name = os.path.basename(dir_name)

                    for frame_number in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        frame_filename = f"{DatasetMap.num_to_str_code[dataset]}_{video_name}_{frame_number:03d}.png"
                        if dataset == Datasets.CasiaFasd.value:
                            if file == "1.avi" or file == "2.avi" or file == "HR_1.avi":
                                frame_filename = (f"{DatasetMap.num_to_str_code[dataset]}_"
                                                  f"client_{client_name}_{video_name}_real_{frame_number:03d}.png")
                            else:
                                frame_filename = (f"{DatasetMap.num_to_str_code[dataset]}"
                                                  f"_client_{client_name}_{video_name}_attack_{frame_number:03d}.png")
                        elif dataset == Datasets.OuluNpu.value:
                            if file.endswith("1.avi"):
                                frame_filename = (f"{DatasetMap.num_to_str_code[dataset]}_{video_name}_"
                                                  f"real_{frame_number:03d}.png")
                            else:
                                frame_filename = (f"{DatasetMap.num_to_str_code[dataset]}_{video_name}_"
                                                  f"attack_{frame_number:03d}.png")

                        # Resize frame if resize_shape is specified
                        if resize_shape is not None:
                            frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)

                        # Save frame as PNG
                        png_path = os.path.join(target_dir, frame_filename)
                        cv2.imwrite(png_path, frame)

                        # Convert PNG to NPY
                        with Image.open(png_path) as img:
                            img_array = np.array(img)
                            npy_filename = frame_filename.replace('.png', '.npy')
                            npy_path = os.path.join(target_dir, npy_filename)
                            np.save(npy_path, img_array)

                        # Remove the intermediate PNG file
                        os.remove(png_path)

                    cap.release()
                    print(f'Processed {video_path}')


# Function to get a random frame from a video
def get_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_number = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    _, frame = cap.read()
    return frame


# Function to create directories and copy random frames
def copy_random_frames(source_dir, destination_dir):
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more video extensions if needed
                video_path = os.path.join(root, filename)
                destination_subdir = os.path.join(destination_dir, os.path.relpath(root, source_dir))
                os.makedirs(destination_subdir, exist_ok=True)
                frame = get_random_frame(video_path)
                frame_filename = os.path.splitext(filename)[0] + '.png'
                frame_path = os.path.join(destination_subdir, frame_filename)
                cv2.imwrite(frame_path, frame)
                print(f"Saved random frame from {filename} to {frame_path}")

