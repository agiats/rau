import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_metadata_files(results_dir):
    """Load all metadata files for different gamma values."""
    gamma_data = {}

    # Find all metadata files
    metadata_files = glob.glob(os.path.join(results_dir, "metadata_gamma_*.json"))

    for file_path in metadata_files:
        # Extract gamma value from filename
        gamma = float(file_path.split("gamma_")[-1].replace(".json", ""))

        with open(file_path, "r") as f:
            data = json.load(f)
            gamma_data[gamma] = data["local_entropy"]

    return gamma_data


def plot_entropy_vs_n(gamma_data, output_dir):
    """Create a plot showing entropy values for each gamma across different n-gram sizes."""
    # Use a more modern style
    plt.style.use("seaborn-v0_8-darkgrid")  # Using a built-in style
    sns.set_style("darkgrid")

    # Create figure with adjusted size for better legend placement
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort gamma values
    gammas = sorted(gamma_data.keys())

    # Get n-gram sizes
    n_gram_sizes = sorted([int(k) for k in list(gamma_data[gammas[0]].keys())])

    # Plot line for each gamma value
    for gamma in gammas:
        entropies = [gamma_data[gamma][str(n)] for n in n_gram_sizes]
        # Format gamma value in scientific notation
        gamma_label = f"γ={gamma:.0e}".replace("e-0", "e-")
        ax.plot(n_gram_sizes, entropies, marker="o", label=gamma_label)

    # Set scales and labels
    ax.set_xlabel("n-gram Size", fontsize=12)
    ax.set_ylabel("Local Entropy", fontsize=12)
    ax.set_title(
        "Local Entropy vs n-gram Size for Different γ Values", fontsize=14, pad=20
    )
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Ensure x-axis ticks are integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Improve legend
    ax.legend(
        title="γ Values",
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=3,
        fontsize=10,
        title_fontsize=12,
    )

    # Adjust layout and save
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_dir / "gamma_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot gamma parameter search results"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory containing the gamma analysis results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the plot will be saved",
    )

    args = parser.parse_args()

    # Load and process the data
    gamma_data = load_metadata_files(args.data_dir)

    if not gamma_data:
        print("No metadata files found!")
    else:
        # Create the plot
        plot_entropy_vs_n(gamma_data, args.output_dir)
        print("Plot saved successfully!")


if __name__ == "__main__":
    main()
