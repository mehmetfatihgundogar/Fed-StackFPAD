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

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re


def determine_label(filename, target):
    """
    Determine the label for each curve based on filename.
    """
    base = filename.replace(".npz", "")
    if base.startswith("central_stacked"):
        return "Centralized w/ stacking"
    if base.startswith("stacked"):
        return "Fed-StackedFPAD w/ stacking"
    if base.startswith("federated_"):
        return "Fed-StackedFPAD w/o stacking"
    if base.startswith("central_"):
        m = re.match(r"central_([A-Z]+)_{}".format(target), base)
        if m:
            train = m.group(1)
            if len(train) == 1:
                return f"Single {train}→{target}"
            else:
                return "Centralized w/o stacking"
    return filename


def parse_title(train_domains, target):
    if len(train_domains) == 1:
        return f"{train_domains} to {target}"
    letters = "&".join(train_domains)
    return f"{letters} to {target}"


def parse_output_filename(train_domains, target):
    if len(train_domains) == 1:
        return f"{train_domains} to {target}.pdf"
    letters = "-".join(train_domains)
    return f"{letters} to {target}.pdf"


def main():
    parser = argparse.ArgumentParser(description="Generate ROC curve plot from npz files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .npz files")
    parser.add_argument("--target", type=str, required=True, help="Target domain letter (e.g., M)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output PDF")
    args = parser.parse_args()

    base_font = 14
    label_font = int(base_font * 1.618)
    title_font = int(label_font * 1.2)
    legend_font = int(base_font * 1.4)

    all_files = os.listdir(args.input_dir)
    run_files = [f for f in all_files if f.endswith(f"_{args.target}.npz") or f.startswith("stacked") or f.startswith("central_stacked")]

    if not run_files:
        print(f"No files found ending with _{args.target}.npz or stacked_*.npz")
        return

    # Determine the longest central train for title
    central_candidates = []
    for f in run_files:
        m = re.match(r"central_([A-Z]+)_{}".format(args.target), f.replace(".npz",""))
        if m:
            train_part = m.group(1)
            central_candidates.append((len(train_part), train_part))

    if central_candidates:
        longest_train = sorted(central_candidates, key=lambda x: -x[0])[0][1]
        title = parse_title(longest_train, args.target)
        output_filename = parse_output_filename(longest_train, args.target)
    else:
        if any(f.startswith("federated_") for f in run_files):
            title = f"Fed-StackedFPAD to {args.target}"
            output_filename = f"Fed-StackedFPAD to {args.target}.pdf"
        elif any(f.startswith("central_stacked") for f in run_files):
            title = f"Centralized w/ stacking to {args.target}"
            output_filename = f"Centralized w- stacking to {args.target}.pdf"
        elif any(f.startswith("stacked") for f in run_files):
            title = f"Fed-StackedFPAD w/ stacking to {args.target}"
            output_filename = f"Fed-StackedFPAD w- stacking to {args.target}.pdf"
        else:
            title = f"ROC Curves to {args.target}"
            output_filename = f"roc_to_{args.target}.pdf"

    output_path = os.path.join(args.output_dir, output_filename)

    # Mapping from label to (fpr, fnr)
    curves = {}

    for run_file in run_files:
        full_path = os.path.join(args.input_dir, run_file)
        data = np.load(full_path)
        fpr = data["fpr"]
        fnr = data["fnr"]
        label = determine_label(run_file, args.target)
        curves[label] = (fpr, fnr)

    # Legend dynamic ordering
    single_labels = sorted([label for label in curves if label.startswith("Single")])
    rest_labels = [
        "Centralized w/o stacking",
        "Centralized w/ stacking",
        "Fed-StackedFPAD w/o stacking",
        "Fed-StackedFPAD w/ stacking"
    ]
    legend_order = single_labels + [l for l in rest_labels if l in curves]

    # Start plot
    plt.figure(figsize=(8,6))
    plt.grid(True, which="major", linestyle="-", linewidth=0.5, color="lightgrey")
    plt.grid(True, which="minor", linestyle="-", linewidth=0.25, color="lightgrey")
    plt.xticks(np.arange(0,1.01,0.2), fontsize=base_font)
    plt.yticks(np.arange(0,1.05,0.2), fontsize=base_font)
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Plot in order
    for label in legend_order:
        fpr, fnr = curves[label]
        plt.plot(fpr, fnr, lw=2, label=label)

    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Acceptance Rate (FAR)", fontsize=label_font)
    plt.ylabel("False Rejection Rate (FRR)", fontsize=label_font)
    plt.title(title, fontsize=title_font, fontweight="bold")
    plt.legend(loc="upper right", fontsize=legend_font, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
