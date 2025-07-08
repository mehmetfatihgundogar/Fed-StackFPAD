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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os


def extract_video_id(name):
    return name.rsplit('_', 1)[0]


def sample_frames(features, labels, ids, max_per_class=200):
    df = pd.DataFrame(features)
    df['label'] = labels
    df['id'] = ids
    df['video_id'] = df['id'].apply(extract_video_id)

    sampled_frames = []
    for class_label in [0, 1]:
        group = df[df['label'] == class_label].groupby('video_id').first().reset_index()
        sampled = group.sample(n=min(len(group), max_per_class), random_state=42)
        sampled_frames.append(sampled)

    df_selected = pd.concat(sampled_frames)
    features_selected = df_selected.drop(columns=['label', 'id', 'video_id']).values
    labels_selected = df_selected['label'].values
    ids_selected = df_selected['id'].values
    return features_selected, labels_selected, ids_selected


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of all domains.")
    parser.add_argument("--target_domain", type=str, required=True, help="Target domain label (e.g., M, I, C, O)")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory containing .npy files for tsne")
    parser.add_argument("--split_type", type=str, required=True, help="Split type (e.g., federated, central)")
    parser.add_argument("--train_set", type=str, required=True, help="Federated training set identifier (e.g., IOC, MOC, IMO, IMC)")
    parser.add_argument("--output_file", type=str, required=True, help="Output PDF filename")

    args = parser.parse_args()

    domains = ["O", "C", "I", "M"]  # it is possible to add new domains
    colors = {"O": "#1f77b4", "C": "#ff7f0e", "I": "#2ca02c", "M": "#d62728"}
    markers = {0: "x", 1: "o"}

    base_font = 14
    label_font = int(base_font * 1.618)
    title_font = int(label_font * 1.2)
    legend_font = int((base_font - 8) * 1.618 * 1.2)

    X_all, y_all, domain_labels = [], [], []

    for domain in domains:
        feature_file = os.path.join(args.feature_dir, f"features_{args.split_type}_{args.train_set}_{domain}.npy")
        label_file = os.path.join(args.feature_dir, f"labels_{args.split_type}_{args.train_set}_{domain}.npy")
        ids_file = os.path.join(args.feature_dir, f"ids_{args.split_type}_{args.train_set}_{domain}.npy")

        X = np.load(feature_file)
        y = np.load(label_file)
        ids = np.load(ids_file, allow_pickle=True)

        X_sel, y_sel, _ = sample_frames(X, y, ids)
        X_all.append(X_sel)
        y_all.append(y_sel)
        domain_labels += [domain] * len(y_sel)

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)
    domain_labels = np.array(domain_labels)

    tsne = TSNE(n_components=2, perplexity=10, learning_rate="auto", random_state=42)
    X_embedded = tsne.fit_transform(X_all)

    plt.figure(figsize=(10, 8))
    for domain in domains:
        for label in [0, 1]:
            idxs = (domain_labels == domain) & (y_all == label)
            domain_type = "target" if domain == args.target_domain else "src"
            class_label = "genuine" if label == 1 else "attack"
            plt.scatter(
                X_embedded[idxs, 0],
                X_embedded[idxs, 1],
                c=colors[domain],
                marker=markers[label],
                label=f"{domain_type}-{domain}-{class_label}",
                alpha=0.9,
                edgecolors="k",
                s=80
            )

    plt.title("t-SNE Visualization of Feature Space", fontsize=title_font, fontweight="bold")
    plt.xticks(fontsize=base_font)
    plt.yticks(fontsize=base_font)
    plt.legend(loc="lower right", fontsize=legend_font, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
