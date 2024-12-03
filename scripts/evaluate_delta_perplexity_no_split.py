import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import scipy.stats as stats
import polars as pl


def calculate_lower_bound_complexity(
    results_dir: Path, dataset_dir: Path, grammar_names: list[str]
) -> list[dict]:
    """Calculate lower bound complexity for each grammar based on sampled sentences."""
    total_count = 100_000_000
    split_num = 10
    lower_bound_complexity = []

    for grammar_name in grammar_names:
        # Load counts file
        counts_file = (
            results_dir
            / "length_sampling"
            / f"100M_samples_eos_zipf_min1_max20{'_' + grammar_name if grammar_name != 'Base' else ''}"
            / "sample_counts.csv.gz"
        )
        counts_df = pl.read_csv(counts_file, new_columns=["sentence", "count"])
        counts_df = counts_df.with_columns(
            (pl.col("count") / total_count).alias("probability")
        )

        # 各splitごとに計算
        for split_i in range(split_num):
            # Load test sentences for this split
            split_file = dataset_dir / grammar_name / f"split_{split_i}.test"
            with open(split_file, "r") as f:
                sentences_in_data = [line.strip() + " [eos]" for line in f.readlines()]

            # Calculate probabilities and entropy for this split
            sentences_in_data = pl.DataFrame({"sentence": sentences_in_data})
            sentences_in_data = sentences_in_data.join(
                counts_df, on="sentence", how="left"
            )
            assert sentences_in_data["probability"].null_count() == 0

            entropy = stats.entropy(sentences_in_data["probability"].to_numpy(), base=2)
            perplexity = 2**entropy
            lower_bound_complexity.append(
                {
                    "grammar_name": grammar_name,
                    "split": split_i,
                    "lower_bound_entropy": entropy,
                    "lower_bound_perplexity": perplexity,
                }
            )

    return lower_bound_complexity


def calculate_model_perplexities(
    results_dir: Path, grammar_names: list[str], model_names: list[str]
) -> pd.DataFrame:
    """Calculate perplexities for each model and grammar combination."""
    num_splits = 10
    result_list = []

    for grammar_name in grammar_names:
        for model_name in model_names:
            model_result_dir = results_dir / f"{model_name}_results"
            grammar_result_dir = model_result_dir / grammar_name

            # 各splitごとに計算
            for split_i in range(num_splits):
                split_result_file = (
                    grammar_result_dir / f"split_{split_i}.test.scores.txt"
                )
                with open(split_result_file) as f:
                    scores = [float(line.strip()) for line in f]

                grammar_probs = [np.exp(score) for score in scores]
                entropy = stats.entropy(grammar_probs, base=2)
                perplexity = 2**entropy

                result_list.append(
                    {
                        "model_name": model_name,
                        "grammar_name": grammar_name,
                        "split": split_i,
                        "entropy": entropy,
                        "perplexity": perplexity,
                    }
                )

    return pd.DataFrame(result_list).sort_values(
        ["model_name", "grammar_name", "split"]
    )


def plot_perplexities(
    result_df: pd.DataFrame,
    grammar_names: list[str],
    model_names: list[str],
    output_file: Path | None = None,
) -> None:
    """Plot perplexities for each model and grammar combination with error bars."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    bar_width = 0.35

    # 各モデル・文法ごとの平均と標準偏差を計算
    stats_df = (
        result_df.groupby(["model_name", "grammar_name"])
        .agg(
            {
                "perplexity": ["mean", "std"],
                "lower_bound_perplexity": "first",  # 下限は全splitで同じ
            }
        )
        .reset_index()
    )
    stats_df.columns = [
        "model_name",
        "grammar_name",
        "perplexity_mean",
        "perplexity_std",
        "lower_bound_perplexity",
    ]

    # Plot raw perplexities with error bars
    for i, model_name in enumerate(model_names):
        model_stats = stats_df[stats_df["model_name"] == model_name]
        ax1.bar(
            np.arange(len(grammar_names)) + i * bar_width,
            model_stats["perplexity_mean"],
            bar_width,
            label=model_name,
            yerr=model_stats["perplexity_std"],
            capsize=5,
        )

    ax1.set_xticks(np.arange(len(grammar_names)) + bar_width / 2)
    ax1.set_xticklabels(grammar_names, rotation=45)
    ax1.set_xlabel("Grammar")
    ax1.set_ylabel("Perplexity")
    ax1.legend()
    ax1.set_title("Raw Perplexities")

    # Plot delta perplexities with error bars
    for i, model_name in enumerate(model_names):
        model_stats = stats_df[stats_df["model_name"] == model_name]
        delta_perplexity = (
            model_stats["perplexity_mean"] - model_stats["lower_bound_perplexity"]
        )
        # 標準偏差はそのまま使用（下限との差分の標準偏差は元の標準偏差と同じ）
        ax2.bar(
            np.arange(len(grammar_names)) + i * bar_width,
            delta_perplexity,
            bar_width,
            label=model_name,
            yerr=model_stats["perplexity_std"],
            capsize=5,
        )

    ax2.set_xticks(np.arange(len(grammar_names)) + bar_width / 2)
    ax2.set_xticklabels(grammar_names, rotation=45)
    ax2.set_xlabel("Grammar")
    ax2.set_ylabel("Perplexity - Lower Bound Perplexity")
    ax2.legend()
    ax2.set_title("Delta Perplexities")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()


def main():
    grammar_names = [
        "Base",
        "DeterministicShuffle",
        "NonDeterministicShuffle",
        "LocalShuffle",
        "EvenOddShuffle",
        "NoReverse",
        "PartialReverse",
        "FullReverse",
    ]
    model_names = ["lstm", "transformer"]

    results_dir = Path("results").resolve()
    dataset_dir = Path("data/fairseq_train/eos_zipf_min1_max_20").resolve()

    # Calculate lower bound complexity
    lower_bound_results = calculate_lower_bound_complexity(
        results_dir, dataset_dir, grammar_names
    )
    lower_bound_results_df = pd.DataFrame(lower_bound_results)

    # Calculate model perplexities
    result_df = calculate_model_perplexities(results_dir, grammar_names, model_names)

    # merge
    results_df = result_df.merge(
        lower_bound_results_df, on=["grammar_name", "split"], how="left"
    )

    # save results
    results_df.to_csv(results_dir / "perplexity_results.csv", index=False)

    # Plot results
    output_file = results_dir / "perplexity_plots.png"
    plot_perplexities(results_df, grammar_names, model_names, output_file)


if __name__ == "__main__":
    main()
