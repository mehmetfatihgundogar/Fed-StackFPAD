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

import video_utils
import csv_utils
from datasets import Datasets, DatasetMap
from typing import List
import os
import shutil
import argparse


base_folder = None
num_of_frame_steps = None

dataset_npy_cache_root_dir = None
experiment_folder_prefix = None

replay_attack_train_folder = None
replay_attack_validation_folder = None
replay_attack_test_folder = None

msu_mfsd_source_folder = None
msu_mfsd_train_sub_list_path = None
msu_mfsd_test_sub_list_path = None

casia_fasd_train_folder = None
casia_fasd_test_folder = None

oulu_npu_train_folder = None
oulu_npu_validation_folder = None
oulu_npu_test_folder = None


def prepare_datasets(is_centralized_setup: bool, training_datasets: List[int], test_dataset):
    num_training_clients = len(training_datasets)
    print(training_datasets)

    dataset_playground_code: str = ''
    dataset_playground_code += experiment_folder_prefix

    if len(training_datasets) > 1:
        if is_centralized_setup:
            dataset_playground_code += 'exp_mode_centralized__'
        else:
            dataset_playground_code += 'exp_mode_multiclient__'
    else:
        dataset_playground_code += 'exp_mode_singleclient__'

    train_dataset_str = ''
    for i in range(len(training_datasets)):
        train_dataset_str += DatasetMap.num_to_str_code[training_datasets[i]]
    dataset_playground_code += str('train_on_' + train_dataset_str + '__')
    dataset_playground_code += str('test_on_' + DatasetMap.num_to_str_code[test_dataset])
    print('Dataset playground code: ', dataset_playground_code)

    prepare_datasets_for_experiment(dataset_playground_code=dataset_playground_code,
                                    num_training_clients=num_training_clients,
                                    training_datasets=training_datasets, test_dataset=test_dataset,
                                    num_of_frame_steps=num_of_frame_steps,
                                    is_centralized_setup=is_centralized_setup)


def prepare_datasets_for_experiment(dataset_playground_code: str, num_training_clients: int,
                                    training_datasets: List[int], test_dataset: int, num_of_frame_steps: int,
                                    is_centralized_setup: bool) -> None:
    destination_folder_for_train: str
    destination_folder_for_test: str
    destination_folder_for_full_dataset: str
    train_sub_list: List[str]
    test_sub_list: List[str]
    cache_folder_for_train: str
    cache_folder_for_test: str
    cache_folder_for_full_dataset: str
    resize_shape = (224, 224)

    # Preprocess msumfsd dataset if required
    cache_step_folder_full = str('step' + str(num_of_frame_steps))
    if (Datasets.MsuMfsd.value in training_datasets
            or test_dataset == Datasets.MsuMfsd.value
            or test_dataset == Datasets.CasiaFasd.value):
        cache_folder_for_full_dataset = os.path.join(dataset_npy_cache_root_dir,
                                                     cache_step_folder_full,
                                                     DatasetMap.num_to_str[Datasets.MsuMfsd.value], 'full')

        if not os.path.isdir(cache_folder_for_full_dataset):
            print('Full data of ', DatasetMap.num_to_str[Datasets.MsuMfsd.value],
                  ' dataset is not found in the cache. Populating data to the cache...')
            video_utils.extract_and_save_frames(msu_mfsd_source_folder, cache_folder_for_full_dataset,
                                                dataset=Datasets.MsuMfsd.value, mode='step', step=num_of_frame_steps,
                                                resize_shape=resize_shape)


    # Prepare training data on cache folder
    cache_step_folder_train = str('step' + str(num_of_frame_steps))
    if Datasets.ReplayAttack.value in training_datasets:
        populate_cache_for_train_data(Datasets.ReplayAttack.value, num_of_frame_steps)

    if Datasets.MsuMfsd.value in training_datasets:
        populate_cache_for_msumfsd_train_data(cache_step_folder_train, cache_folder_for_full_dataset)

    if Datasets.CasiaFasd.value in training_datasets:
        populate_cache_for_train_data(Datasets.CasiaFasd.value, num_of_frame_steps)

    if Datasets.OuluNpu.value in training_datasets:
        populate_cache_for_train_data(Datasets.OuluNpu.value, num_of_frame_steps)

    # Copy train data from cache to the experiment folder
    destination_folder_for_train = os.path.join(base_folder, dataset_playground_code, 'train')
    for i in range(len(training_datasets)):
        print('Copying train fold of ', DatasetMap.num_to_str[training_datasets[i]],
              ' dataset from cache to the experiment folder...')
        cache_folder_for_train = os.path.join(dataset_npy_cache_root_dir,
                                              cache_step_folder_train,
                                              DatasetMap.num_to_str[training_datasets[i]], 'train')
        shutil.copytree(cache_folder_for_train, destination_folder_for_train, dirs_exist_ok=True)
        print('Train fold of ', DatasetMap.num_to_str[training_datasets[i]],
              ' dataset is copied to the experiment folder...')

    # Prepare validation data
    cache_step_folder_validation = str('step' + str(num_of_frame_steps))
    validation_dataset = test_dataset
    cache_folder_for_validation = os.path.join(dataset_npy_cache_root_dir,
                                               cache_step_folder_validation,
                                               DatasetMap.num_to_str[validation_dataset], 'validation')
    destination_folder_for_validation = os.path.join(base_folder, dataset_playground_code, 'validation')
    if validation_dataset == Datasets.ReplayAttack.value:
        populate_cache_for_validation_data(validation_dataset, num_of_frame_steps)
    elif validation_dataset == Datasets.MsuMfsd.value:
        populate_cache_for_msumfsd_train_data(cache_step_folder_train, cache_folder_for_full_dataset)
    elif validation_dataset == Datasets.CasiaFasd.value:
        populate_cache_for_train_data(Datasets.CasiaFasd.value, num_of_frame_steps)
    elif validation_dataset == Datasets.OuluNpu.value:
        populate_cache_for_validation_data(validation_dataset, num_of_frame_steps)
    else:
        print('Not implemented yet!')

    if validation_dataset == Datasets.MsuMfsd.value or validation_dataset == Datasets.CasiaFasd.value:
        source_folder_for_validation = os.path.join(dataset_npy_cache_root_dir,
                                              cache_step_folder_train,
                                              DatasetMap.num_to_str[validation_dataset], 'train')
        shutil.copytree(source_folder_for_validation, destination_folder_for_validation)
        print('Validation fold of ', DatasetMap.num_to_str[validation_dataset],
              ' dataset is copied to the experiment folder from its train fold.')
    else:
        shutil.copytree(cache_folder_for_validation, destination_folder_for_validation)
        print('Validation fold of ', DatasetMap.num_to_str[validation_dataset],
              ' dataset is copied to the experiment folder.')

    # Prepare test data
    cache_step_folder_test = str('step' + str(num_of_frame_steps))
    cache_folder_for_test = os.path.join(dataset_npy_cache_root_dir,
                                         cache_step_folder_test,
                                         DatasetMap.num_to_str[test_dataset], 'test')
    destination_folder_for_test = os.path.join(base_folder, dataset_playground_code, 'test')
    if test_dataset == Datasets.ReplayAttack.value:
        populate_cache_for_test_data(test_dataset, num_of_frame_steps)
    elif test_dataset == Datasets.MsuMfsd.value:
        if not os.path.isdir(cache_folder_for_test):
            print('Test fold of ', DatasetMap.num_to_str[test_dataset],
                  ' dataset is not found in the cache. Populating data to the cache...')
            with open(msu_mfsd_test_sub_list_path, 'r') as file:
                test_sub_list = ['client' + line.strip().zfill(3) for line in file]
            move_files(test_sub_list, cache_folder_for_full_dataset, cache_folder_for_test)
    elif test_dataset == Datasets.CasiaFasd.value:
        populate_cache_for_test_data(test_dataset, num_of_frame_steps)
    elif test_dataset == Datasets.OuluNpu.value:
        populate_cache_for_test_data(test_dataset, num_of_frame_steps)
    else:
        print('Not implemented yet!')

    shutil.copytree(cache_folder_for_test, destination_folder_for_test)
    print('Test fold of ', DatasetMap.num_to_str[test_dataset],
          ' dataset is copied to the experiment folder...')

    assert len(destination_folder_for_train) != 0
    assert len(destination_folder_for_validation) != 0
    assert len(destination_folder_for_test) != 0

    # Prepare CSV files
    if num_training_clients == 1 or is_centralized_setup:
        os.mkdir(os.path.join(base_folder, dataset_playground_code, 'central'))
        destination_path_for_train_csv = os.path.join(base_folder, dataset_playground_code, 'central', 'train.csv')
        csv_utils.list_filenames_to_csv(directory=destination_folder_for_train,
                                        output_csv=destination_path_for_train_csv)
    else:
        os.makedirs(os.path.join(base_folder, dataset_playground_code,
                                 str(num_training_clients) + '_clients', 'federated'))
        for i in range(len(training_datasets)):
            cache_folder_for_train = os.path.join(dataset_npy_cache_root_dir,
                                                  cache_step_folder_train,
                                                  DatasetMap.num_to_str[training_datasets[i]], 'train')
            destination_path_for_train_csv = os.path.join(base_folder, dataset_playground_code,
                                                          str(num_training_clients) + '_clients', 'federated',
                                                          f'client_{i + 1}.csv')
            csv_utils.list_filenames_to_csv(directory=cache_folder_for_train,
                                            output_csv=destination_path_for_train_csv)

    csv_utils.list_filenames_to_csv(directory=destination_folder_for_validation,
                                    output_csv=str(destination_folder_for_validation + ".csv"))

    csv_utils.list_filenames_to_csv(directory=destination_folder_for_test,
                                    output_csv=str(destination_folder_for_test + ".csv"))

    csv_utils.generate_labels_csv(root_dir=os.path.join(base_folder, dataset_playground_code),
                                  output_csv=os.path.join(base_folder, dataset_playground_code, "labels.csv"))


def move_files(sub_list, source_folder, destination_folder):
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if any(sub in filename for sub in sub_list):
            shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))


def populate_cache_for_train_data(train_dataset, num_steps_train):
    cache_step_folder_train = str('step' + str(num_steps_train))
    cache_folder_for_train = os.path.join(dataset_npy_cache_root_dir,
                                          cache_step_folder_train,
                                          DatasetMap.num_to_str[train_dataset], 'train')

    resize_shape = (224, 224)
    source_folder = ''
    if train_dataset == Datasets.ReplayAttack.value:
        source_folder = replay_attack_train_folder
    elif train_dataset == Datasets.CasiaFasd.value:
        source_folder = casia_fasd_train_folder
    elif train_dataset == Datasets.OuluNpu.value:
        source_folder = oulu_npu_train_folder

    if not os.path.isdir(cache_folder_for_train):
        print('Train fold of ', DatasetMap.num_to_str[train_dataset],
              ' dataset is not found in the cache. Populating data to the cache...')
        video_utils.extract_and_save_frames(source_folder, cache_folder_for_train,
                                            dataset=train_dataset, mode='step', step=num_steps_train,
                                            resize_shape=resize_shape)


def populate_cache_for_msumfsd_train_data(cache_step_folder_train, cache_folder_for_full_dataset):
    cache_folder_for_train = os.path.join(dataset_npy_cache_root_dir,
                                          cache_step_folder_train,
                                          DatasetMap.num_to_str[Datasets.MsuMfsd.value], 'train')
    if not os.path.isdir(cache_folder_for_train):
        print('Train fold of ', DatasetMap.num_to_str[Datasets.MsuMfsd.value],
              ' dataset is not found in the cache. Populating data to the cache...')
        with open(msu_mfsd_train_sub_list_path, 'r') as file:
            train_sub_list = ['client' + line.strip().zfill(3) for line in file]
        move_files(train_sub_list, cache_folder_for_full_dataset, cache_folder_for_train)


def populate_cache_for_validation_data(validation_dataset, num_steps_validation):
    cache_step_folder_validation = str('step' + str(num_steps_validation))
    cache_folder_for_validation = os.path.join(dataset_npy_cache_root_dir,
                                               cache_step_folder_validation,
                                               DatasetMap.num_to_str[validation_dataset], 'validation')
    resize_shape = (224, 224)

    source_folder = ''
    if validation_dataset == Datasets.ReplayAttack.value:
        source_folder = replay_attack_validation_folder
    elif validation_dataset == Datasets.OuluNpu.value:
        source_folder = oulu_npu_validation_folder

    if not os.path.isdir(cache_folder_for_validation):
        print('Validation fold of ', DatasetMap.num_to_str[validation_dataset],
              ' dataset is not found in the cache. Populating data to the cache...')
        video_utils.extract_and_save_frames(source_folder, cache_folder_for_validation,
                                            dataset=validation_dataset, mode='step', step=num_steps_validation,
                                            resize_shape=resize_shape)


def populate_cache_for_test_data(test_dataset, num_steps_test):
    cache_step_folder_test = str('step' + str(num_steps_test))
    cache_folder_for_test = os.path.join(dataset_npy_cache_root_dir,
                                         cache_step_folder_test,
                                         DatasetMap.num_to_str[test_dataset], 'test')
    resize_shape = (224, 224)

    source_folder = ''
    if test_dataset == Datasets.ReplayAttack.value:
        source_folder = replay_attack_test_folder
    elif test_dataset == Datasets.CasiaFasd.value:
        source_folder = casia_fasd_test_folder
    elif test_dataset == Datasets.OuluNpu.value:
        source_folder = oulu_npu_test_folder

    if not os.path.isdir(cache_folder_for_test):
        print('Test fold of ', DatasetMap.num_to_str[test_dataset],
              ' dataset is not found in the cache. Populating data to the cache...')
        video_utils.extract_and_save_frames(source_folder, cache_folder_for_test,
                                            dataset=test_dataset, mode='step', step=num_steps_test,
                                            resize_shape=resize_shape)


def main():
    DATASET_CHOICES = {d.value: d.name for d in Datasets}

    def parse_training_datasets(values: str):
        items = values.split(',')
        dataset_ids = []
        for item in items:
            try:
                dataset_id = int(item)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid dataset id '{item}'. Must be integer.")
            if dataset_id not in DATASET_CHOICES:
                raise argparse.ArgumentTypeError(
                    f"Invalid dataset id '{dataset_id}'. Allowed: "
                    + ", ".join([f"{k}:{v}" for k, v in DATASET_CHOICES.items()])
                )
            dataset_ids.append(dataset_id)
        if len(dataset_ids) not in [1, 2, 3, 4]:
            raise argparse.ArgumentTypeError("Training dataset list must contain 1, 2, or 3 IDs.")
        return dataset_ids

    parser = argparse.ArgumentParser(
        description="Prepare datasets for Fed-StackFPAD.\n\n"
                    "Available Dataset IDs:\n" +
                    "\n".join([f"  {id}: {name}" for id, name in DATASET_CHOICES.items()]),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base_folder", type=str, required=True)
    parser.add_argument("--num_of_frame_steps", type=int, required=True)
    parser.add_argument("--is_centralized_setup", action="store_true")
    parser.add_argument("--training_datasets", type=parse_training_datasets, required=True)
    parser.add_argument("--test_dataset", type=int, choices=list(DATASET_CHOICES.keys()), required=True)

    args = parser.parse_args()

    global base_folder
    global num_of_frame_steps
    global dataset_npy_cache_root_dir
    global experiment_folder_prefix
    global replay_attack_train_folder
    global replay_attack_validation_folder
    global replay_attack_test_folder
    global msu_mfsd_source_folder
    global msu_mfsd_train_sub_list_path
    global msu_mfsd_test_sub_list_path
    global casia_fasd_train_folder
    global casia_fasd_test_folder
    global oulu_npu_train_folder
    global oulu_npu_validation_folder
    global oulu_npu_test_folder

    base_folder = args.base_folder
    num_of_frame_steps = args.num_of_frame_steps

    dataset_npy_cache_root_dir = os.path.join(base_folder, 'cache-npy')
    experiment_folder_prefix = 'fed-stackfpad__'

    replay_attack_train_folder = [os.path.join(base_folder, "replayattack/train")]
    replay_attack_validation_folder = [os.path.join(base_folder, "replayattack/devel")]
    replay_attack_test_folder = [os.path.join(base_folder, "replayattack/test")]

    msu_mfsd_source_folder = [os.path.join(base_folder, "MSU-MFSD/scene01")]
    msu_mfsd_train_sub_list_path = os.path.join(base_folder, "MSU-MFSD/train_sub_list.txt")
    msu_mfsd_test_sub_list_path = os.path.join(base_folder, "MSU-MFSD/test_sub_list.txt")

    casia_fasd_train_folder = [os.path.join(base_folder, "casiafasd/train_release")]
    casia_fasd_test_folder = [os.path.join(base_folder, "casiafasd/test_release")]

    oulu_npu_train_folder = [os.path.join(base_folder, "oulunpu/Train_files")]
    oulu_npu_validation_folder = [os.path.join(base_folder, "oulunpu/Dev_files")]
    oulu_npu_test_folder = [os.path.join(base_folder, "oulunpu/Test_files")]

    print("==== Selected Dataset IDs ====")
    for tid in args.training_datasets:
        print(f"Training Dataset: {tid} ({DATASET_CHOICES[tid]})")
    print(f"Test Dataset: {args.test_dataset} ({DATASET_CHOICES[args.test_dataset]})")
    print("==============================")

    prepare_datasets(
        is_centralized_setup=args.is_centralized_setup,
        training_datasets=args.training_datasets,
        test_dataset=args.test_dataset
    )


if __name__ == "__main__":
    main()

