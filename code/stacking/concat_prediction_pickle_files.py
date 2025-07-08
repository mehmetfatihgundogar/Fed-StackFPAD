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

import pickle
import numpy as np
import pandas as pd
import os
import argparse


def load_and_sort_predictions(file_pattern, num_files, target):
    """
    Loads prediction files and sorts them based on names.

    Args:
        file_pattern (str): The file path pattern to load data.
        num_files (int): Number of prediction files to load.
        target (str): Target dataset name.

    Returns:
        np.ndarray: Stacked prediction features with true labels.
        list: Corresponding sorted sample names.
    """
    prediction_dfs = []
    reference_names = None
    true_labels = None

    for i in range(1, num_files + 1):
        filepath = file_pattern.format(target=target, i=i)
        with open(filepath, "rb") as file:
            data = pickle.load(file)

            df = pd.DataFrame({
                "names": data["names"],
                f"pred_{i}": data["pred"]
            })

            prediction_dfs.append(df)

            if reference_names is None:
                reference_names = df[["names"]]
                true_labels = pd.DataFrame({"names": data["names"], "true_labels": data["true_labels"]})

    final_df = reference_names.copy()
    for df in prediction_dfs:
        final_df = final_df.merge(df, on="names", how="left")

    final_df = final_df.merge(true_labels, on="names", how="left")
    final_df = final_df.sort_values("names").reset_index(drop=True)

    sorted_names = final_df["names"].tolist()
    sorted_true_labels = final_df["true_labels"].values
    sorted_predictions = final_df.iloc[:, 1:-1].values

    stacked_features = np.column_stack([sorted_predictions, sorted_true_labels])

    return stacked_features, sorted_names


def main():
    parser = argparse.ArgumentParser(
        description="Prepare stacked validation and test data for Fed-StackFPAD."
    )
    parser.add_argument( "--target", type=str, required=True,
                         help="Target dataset name (e.g., I, M, O, C)"
    )
    parser.add_argument("--num_files", type=int, required=True,
        help="Number of prediction files to merge (e.g., 4)"
    )
    parser.add_argument("--prediction_data_folder", type=str, required=True,
        help="Directory containing prediction pickle files"
    )

    args = parser.parse_args()

    target = args.target
    num_files = args.num_files
    prediction_data_folder = args.prediction_data_folder

    # Validation data
    validation_file_pattern = os.path.join(
        prediction_data_folder, "prediction_data_validation_target_{target}_{i}.pkl"
    )
    validation_features, validation_names = load_and_sort_predictions(
        validation_file_pattern, num_files, target
    )

    validation_output = os.path.join(
        prediction_data_folder, f"validation_data_target_{target}.pkl"
    )
    with open(validation_output, "wb") as file:
        pickle.dump({"features": validation_features, "names": validation_names}, file)

    print(f"Validation data saved to: {validation_output}")

    # Test data
    test_file_pattern = os.path.join(
        prediction_data_folder, "prediction_data_test_target_{target}_{i}.pkl"
    )
    test_features, test_names = load_and_sort_predictions(
        test_file_pattern, num_files, target
    )

    test_output = os.path.join(
        prediction_data_folder, f"test_data_target_{target}.pkl"
    )
    with open(test_output, "wb") as file:
        pickle.dump({"features": test_features, "names": test_names}, file)

    print(f"Test data saved to: {test_output}")


if __name__ == "__main__":
    main()
