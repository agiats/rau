import matplotlib

matplotlib.use("Agg")  # Add this before importing pyplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from scipy import stats
import re


def get_experiment_name(x):
    """Extract experiment name from grammar name."""
    return x.split("_")[0]


def get_seed(x):
    """Extract seed from grammar name if it exists."""
    if "seed" in x:
        return int(re.search(r"seed(\d+)", x).group(1))
    return None


def get_window(x):
    """Extract window size from grammar name if it exists."""
    if "window" in x:
        return int(re.search(r"window(\d+)", x).group(1))
    return None


def plot_local_entropy_comparison(df):
    """Plot local entropy comparison across different experimental conditions."""
    color_map = {
        "Base": "red",
        "EvenOddShuffle": "darkgreen",
        "Reverse": "purple",
        "DeterministicShuffle": "yellow",
    }

    n_grams = [
        col.split("_")[0] for col in df.columns if col.endswith("_local_entropy")
    ]
    n_plots = len(n_grams)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    experiment_order = [
        "LocalShuffle",
        "DeterministicShuffle",
        "EvenOddShuffle",
        "Reverse",
    ]

    for i, n_gram in enumerate(n_grams):
        base_value = df[df["experiment_name"] == "Base"][
            f"{n_gram}_local_entropy"
        ].values[0]

        for exp_name in experiment_order:
            exp_data = df[df["experiment_name"] == exp_name]

            if exp_name == "LocalShuffle":
                sns.scatterplot(
                    data=exp_data,
                    x="experiment_name",
                    y=f"{n_gram}_local_entropy",
                    hue="window",
                    ax=axes[i],
                    palette="coolwarm",
                    s=80,
                    alpha=0.6,
                )
            elif exp_name == "DeterministicShuffle":
                axes[i].scatter(
                    [exp_name] * len(exp_data),
                    exp_data[f"{n_gram}_local_entropy"],
                    color="gray",
                    s=80,
                    alpha=0.6,
                    label="DeterministicShuffle",
                )
            else:
                axes[i].scatter(
                    [exp_name] * len(exp_data),
                    exp_data[f"{n_gram}_local_entropy"],
                    color=color_map[exp_name],
                    s=80,
                    label=exp_name,
                )

        axes[i].axhline(y=base_value, color="red", linestyle="--", alpha=0.6)
        axes[i].set_title(f"{n_gram}-local Entropy")
        axes[i].set_ylabel("Local Entropy (bits)" if i == 0 else "")
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis="x", rotation=20)

        if i != 0:
            axes[i].get_legend().remove()

    plt.suptitle("m-local entropy estimated by m-gram (bits)")
    plt.tight_layout()

    return fig


def plot_cross_entropy_correlation(df, architecture_map):
    """Plot correlation between local entropy and cross entropy for each architecture."""
    n_grams = [
        col.split("_")[0] for col in df.columns if col.endswith("_local_entropy")
    ]
    marker_map = {
        "LocalShuffle": "o",
        "DeterministicShuffle": "s",
        "Reverse": "^",
        "EvenOddShuffle": "D",
        "Base": "*",
    }
    color_map = {
        "Base": "red",
        "EvenOddShuffle": "darkgreen",
        "Reverse": "purple",
        "DeterministicShuffle": "orange",
    }

    for architecture in df["architecture"].unique():
        model_data = df[df["architecture"] == architecture]
        fig, axes = plt.subplots(1, len(n_grams), figsize=(5 * len(n_grams), 5))

        for i, n_gram in enumerate(n_grams):
            # Plot base experiments
            for exp_name in ["Base", "EvenOddShuffle", "Reverse"]:
                exp_data = model_data[model_data["experiment_name"] == exp_name]
                if not exp_data.empty:
                    # Average across training seeds for each grammar
                    exp_data_mean = (
                        exp_data.groupby("grammar_name")
                        .agg(
                            {
                                f"{n_gram}_local_entropy": "mean",
                                "cross_entropy_per_token_base_2": "mean",
                            }
                        )
                        .reset_index()
                    )

                    axes[i].scatter(
                        exp_data_mean[f"{n_gram}_local_entropy"],
                        exp_data_mean["cross_entropy_per_token_base_2"],
                        marker=marker_map[exp_name],
                        color=color_map[exp_name],
                        s=150 if exp_name == "Base" else 80,
                        alpha=0.6,
                        label=exp_name,
                    )

            # Plot DeterministicShuffle
            det_data = model_data[
                model_data["experiment_name"] == "DeterministicShuffle"
            ]
            if not det_data.empty:
                det_data_mean = (
                    det_data.groupby("grammar_name")
                    .agg(
                        {
                            f"{n_gram}_local_entropy": "mean",
                            "cross_entropy_per_token_base_2": "mean",
                        }
                    )
                    .reset_index()
                )

                axes[i].scatter(
                    det_data_mean[f"{n_gram}_local_entropy"],
                    det_data_mean["cross_entropy_per_token_base_2"],
                    marker=marker_map["DeterministicShuffle"],
                    color=color_map["DeterministicShuffle"],
                    s=80,
                    alpha=0.6,
                    edgecolor="white",
                    linewidth=0.5,
                    label="DeterministicShuffle",
                )

            # Plot LocalShuffle
            local_data = model_data[
                model_data["experiment_name"].str.contains("LocalShuffle", na=False)
            ]
            if not local_data.empty:
                # Average across training seeds for each window size
                local_data_mean = (
                    local_data.groupby(["window", "grammar_name"])
                    .agg(
                        {
                            f"{n_gram}_local_entropy": "mean",
                            "cross_entropy_per_token_base_2": "mean",
                        }
                    )
                    .reset_index()
                )

                sns.scatterplot(
                    data=local_data_mean,
                    x=f"{n_gram}_local_entropy",
                    y="cross_entropy_per_token_base_2",
                    hue="window",
                    marker=marker_map["LocalShuffle"],
                    ax=axes[i],
                    palette="coolwarm",
                    s=80,
                )

            # Add regression line and R² value
            all_data = pd.concat(
                [
                    model_data[model_data["experiment_name"] == exp_name][
                        [f"{n_gram}_local_entropy", "cross_entropy_per_token_base_2"]
                    ]
                    for exp_name in [
                        "Base",
                        "EvenOddShuffle",
                        "Reverse",
                        "DeterministicShuffle",
                    ]
                ]
                + [
                    local_data_mean[
                        [f"{n_gram}_local_entropy", "cross_entropy_per_token_base_2"]
                    ]
                ]
                if not local_data.empty
                else []
            )

            if not all_data.empty:
                x = all_data[f"{n_gram}_local_entropy"]
                y = all_data["cross_entropy_per_token_base_2"]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                line_x = np.array([x.min(), x.max()])
                line_y = slope * line_x + intercept
                axes[i].plot(line_x, line_y, color="red", linestyle="--", alpha=0.3)
                axes[i].text(
                    0.05,
                    0.95,
                    f"R² = {r_value**2:.3f}",
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    fontsize=14,
                )

            axes[i].set_xlabel(f"{n_gram}-local Entropy", fontsize=16)
            axes[i].set_ylabel("Cross Entropy" if i == 0 else "", fontsize=16)
            axes[i].grid(True, alpha=0.3)

            # Get legend handles and labels
            handles, labels = axes[i].get_legend_handles_labels()
            if not local_data.empty:
                n_windows = len(local_data["window"].unique())
                window_labels = [
                    f"LocalShuffle (k={w})"
                    for w in sorted(local_data["window"].unique())
                ]
                labels = labels[:-n_windows] + window_labels

            if i > 0:
                axes[i].yaxis.set_ticklabels([])
                if axes[i].get_legend() is not None:
                    axes[i].get_legend().remove()
            else:
                # 左端の図のみlegendを表示し、位置を右下に
                axes[i].legend(
                    handles,
                    labels,
                    loc="lower right",  # 右下に配置
                    fontsize=8,  # フォントサイズを小さく
                    markerscale=0.7,  # マーカーサイズを小さく
                    handletextpad=0.5,  # マーカーとテキストの間隔を小さく
                    labelspacing=0.5,  # 凡例の項目間の間隔を小さく
                    bbox_to_anchor=(1.0, 0.0),  # 右下隅に配置
                )

        plt.suptitle(f"{architecture_map[architecture]}", fontsize=20, y=0.9)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        yield fig


def create_latex_table(df):
    """Create LaTeX table of local entropy statistics."""
    local_shuffle_stats = (
        df[df["grammar_name"].str.contains("LocalShuffle")]
        .agg(
            {
                "2_local_entropy": ["mean", "std"],
                "3_local_entropy": ["mean", "std"],
                "4_local_entropy": ["mean", "std"],
                "5_local_entropy": ["mean", "std"],
            }
        )
        .round(3)
    )

    other_exps_mask = df["grammar_name"].str.contains(
        "Base|EvenOddShuffle|Reverse|DeterministicShuffle"
    )
    other_exps = (
        df[other_exps_mask]
        .groupby(df["grammar_name"].apply(lambda x: x.split("_")[0]))[
            ["2_local_entropy", "3_local_entropy", "4_local_entropy", "5_local_entropy"]
        ]
        .mean()
        .round(3)
    )

    local_shuffle_row = pd.DataFrame(
        {
            col: f"{local_shuffle_stats[col]['mean']} ± {local_shuffle_stats[col]['std']}"
            for col in [
                "2_local_entropy",
                "3_local_entropy",
                "4_local_entropy",
                "5_local_entropy",
            ]
        },
        index=["LocalShuffle"],
    )

    latex_df = pd.concat([local_shuffle_row, other_exps])
    latex_df.columns = ["2-local", "3-local", "4-local", "5-local"]

    return latex_df.to_latex(
        escape=False,
        caption="M-local entropy values for different experimental conditions. LocalShuffle values are shown as mean ± standard deviation.",
        label="tab:local_entropy",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot natural language analysis results"
    )
    parser.add_argument(
        "--results_path", type=str, required=True, help="Path to the CSV results file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Directory to save the plots"
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        default=["transformer", "lstm"],
        help="List of architectures to plot",
    )
    parser.add_argument(
        "--architecture_labels",
        type=str,
        nargs="+",
        default=["Transformer", "LSTM"],
        help="Display labels for architectures",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Name of the split to plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture_map = dict(zip(args.architectures, args.architecture_labels))

    # Read and preprocess data
    df = pd.read_csv(args.results_path)
    df["experiment_name"] = df["grammar_name"].apply(get_experiment_name)
    df["seed"] = df["grammar_name"].apply(get_seed)
    df["window"] = df["grammar_name"].apply(get_window)

    # Create and save plots
    comparison_fig = plot_local_entropy_comparison(df)
    comparison_fig.savefig(output_dir / "local_entropy_comparison.png")
    plt.close(comparison_fig)

    # Create and save correlation plots
    for i, fig in enumerate(plot_cross_entropy_correlation(df, architecture_map)):
        arch = list(df["architecture"].unique())[i]
        fig.savefig(
            output_dir
            / f"local_entropy_cross_entropy_correlation_{arch}_{args.split_name}.png"
        )
        plt.close(fig)

    # Create and save LaTeX table
    with open(output_dir / f"local_entropy_table_{args.split_name}.tex", "w") as f:
        f.write(create_latex_table(df))

    plt.close("all")


if __name__ == "__main__":
    main()
