import sys

sys.path.append("..")
from src.length_sampling.sampler import construct_pcfg_sampler
from src.length_sampling.grammars.pcfg import Grammar
from src.length_sampling.grammars.cfg import Nonterminal
from src.length_sampling.util import group_by, get_random_generator_and_seed
import argparse
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import json
import gzip
from src.length_sampling.lower_bound_perplexity import (
    parts_to_perplexity,
    Parts,
)
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy


def plot_probability_distributions(
    data, comparison_name, comparison_prob, figsize=(12, 6), output_path=None
):
    """
    Plot true probability distribution with one comparison distribution

    Args:
        data (pd.DataFrame): DataFrame containing true_prob
        comparison_name (str): Name of the comparison distribution
        comparison_prob (np.array): Probability values of the comparison distribution
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)

    # Create bins for histogram
    bins = np.logspace(
        np.log10(data["true_prob"].min()), np.log10(data["true_prob"].max()), num=30
    )

    # Plot both distributions
    plt.hist(
        data["true_prob"],
        bins=bins,
        log=True,
        alpha=0.5,
        color="#2E86C1",
        edgecolor="white",
        linewidth=0.5,
        label="True Probability",
    )

    plt.hist(
        comparison_prob,
        bins=bins,
        log=True,
        alpha=0.5,
        color="#E74C3C",
        edgecolor="white",
        linewidth=0.5,
        label=comparison_name,
    )

    plt.xscale("log")
    plt.title(f"True Probability vs {comparison_name} (Log Scale)", fontsize=12, pad=15)
    plt.xlabel("Probability (log scale)", fontsize=10)
    plt.ylabel("Frequency (log scale)", fontsize=10)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    if output_path:
        plt.savefig(output_path)


def plot_sentence_count_distribution(df, output_path):
    """
    Plots the distribution of sentence counts on a log scale.

    Args:
        df (pd.DataFrame): DataFrame containing the 'count' column.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))

    plt.hist(
        df["count"],
        log=True,
        alpha=0.7,
        color="#2E86C1",
        edgecolor="white",
        linewidth=0.5,
    )
    plt.xscale("log")

    plt.title("Distribution of Sentence Counts (Log Scale)", fontsize=12, pad=15)
    plt.xlabel("Count (log scale)", fontsize=10)
    plt.ylabel("Frequency (log scale)", fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.1)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    if output_path:
        plt.savefig(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=lambda p: Path(p).resolve())
    parser.add_argument("--input_suffix", type=str, default="json.gz")
    parser.add_argument("--sample_size", type=int, default=-1)
    args = parser.parse_args()

    print(f"Reading sentences from {args.exp_dir}...")
    dfs = []
    for file in args.exp_dir.glob(f"*{args.input_suffix}"):
        with gzip.open(file, "rt") as f:
            dfs.append(pd.read_json(f, lines=True, orient="records"))

    print(f"{len(dfs)} files found.")
    df = pd.concat(dfs, ignore_index=True)

    # distribution of counts
    plot_sentence_count_distribution(
        df, output_path=args.exp_dir / "sampled_sentence_count_distribution.png"
    )

    df["estimated_prob"] = df["count"] / args.sample_size
    kl_div_monte_carlo = entropy(df["true_prob"], df["estimated_prob"])

    print(f"KL Divergence (Monte Carlo): {kl_div_monte_carlo:.6f}")
    # save
    with open(args.exp_dir / "kl_divergence_between_true_and_estimated_dist.value", "w") as f:
        f.write(str(kl_div_monte_carlo))

    plot_probability_distributions(
        df,
        "Estimated Probability",
        df["estimated_prob"],
        output_path=args.exp_dir / "probability_distributions.png",
    )


if __name__ == "__main__":
    main()
