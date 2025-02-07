import matplotlib

matplotlib.use("Agg")  # Add this before importing pyplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from scipy import stats


def plot_entropy_distributions(df):
    """Plot local and global entropy distributions."""
    # Prepare data
    local_entropy_cols = [f"{m}_local_entropy" for m in [2, 3, 4, 5]]
    local_entropy_data = df[local_entropy_cols]
    global_entropy_data = df[["next_symbol_entropy"]]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot local entropy distributions
    sns.swarmplot(data=local_entropy_data, ax=ax1, size=8, alpha=0.6)
    sns.boxplot(
        data=local_entropy_data,
        ax=ax1,
        color="white",
        width=0.3,
        showfliers=False,
        boxprops={"alpha": 0.5},
    )

    # Plot global entropy distribution
    sns.swarmplot(data=global_entropy_data, ax=ax2, size=8, alpha=0.6)
    sns.boxplot(
        data=global_entropy_data,
        ax=ax2,
        color="white",
        width=0.3,
        showfliers=False,
        boxprops={"alpha": 0.5},
    )

    # Customize plots
    ax1.set_xticklabels(
        ["2-local Entropy", "3-local Entropy", "4-local Entropy", "5-local Entropy"],
        fontsize=12,
        color="black",
    )
    ax1.set_ylabel("Entropy Value", fontsize=12)
    ax1.set_title("Local Entropy Distributions", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.set_xticklabels(["Global Entropy"], fontsize=12)
    ax2.set_ylabel("Entropy Value", fontsize=12)
    ax2.set_title("Global Entropy Distribution", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print statistics
    print("\nLocal Entropy Statistics:")
    print(local_entropy_data.describe())
    print("\nGlobal Entropy Statistics:")
    print(global_entropy_data.describe())

    return fig


def plot_local_entropy_vs_cross_entropy(df, architecture_map):
    """Plot local entropy vs cross entropy with regression lines for each architecture."""
    # Get m values from column names
    ms = [
        int(col.replace("_local_entropy", ""))
        for col in df.columns
        if (col.endswith("_local_entropy") and not col.startswith("estimated_"))
    ]

    # Plot for each architecture
    for architecture in df["architecture"].unique():
        # Get data for this architecture
        model_data = df[df["architecture"] == architecture]

        # Create subplots
        fig, axes = plt.subplots(1, len(ms), figsize=(5 * len(ms), 5))

        for i, m in enumerate(ms):
            # Average across training seeds
            model_data_mean = (
                model_data.groupby("grammar_name")
                .agg(
                    {
                        f"{m}_local_entropy": "mean",
                        "cross_entropy_per_token_base_2": "mean",
                    }
                )
                .reset_index()
            )

            sns.scatterplot(
                data=model_data_mean,
                x=f"{m}_local_entropy",
                y="cross_entropy_per_token_base_2",
                ax=axes[i],
                s=100,
                color="blue",
            )

            # Add regression line
            x = model_data_mean[f"{m}_local_entropy"]
            y = model_data_mean["cross_entropy_per_token_base_2"]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            axes[i].plot(line_x, line_y, color="red", linestyle="--", alpha=0.3)

            # Add R² value
            r_squared = r_value**2
            axes[i].text(
                0.05,
                0.95,
                f"R² = {r_squared:.3f}",
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=14,
            )

            axes[i].set_xlabel(f"{m}-local Entropy", fontsize=16)
            axes[i].set_ylabel("Cross Entropy" if i == 0 else "", fontsize=16)
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(f"{architecture_map[architecture]}", fontsize=20, y=0.9)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        yield fig


def plot_local_entropy_vs_kl_divergence(df, architecture_map):
    """Plot local entropy vs KL divergence with regression lines for each architecture."""
    # Get m values from column names
    ms = [
        int(col.replace("_local_entropy", ""))
        for col in df.columns
        if (col.endswith("_local_entropy") and not col.startswith("estimated_"))
    ]

    # Plot for each architecture
    for architecture in df["architecture"].unique():
        # Get data for this architecture
        model_data = df[df["architecture"] == architecture]

        # Create subplots
        fig, axes = plt.subplots(1, len(ms), figsize=(5 * len(ms), 5))

        for i, m in enumerate(ms):
            # Average across training seeds
            model_data_mean = (
                model_data.groupby("grammar_name")
                .agg(
                    {
                        f"{m}_local_entropy": "mean",
                        "KL_divergence": "mean",
                    }
                )
                .reset_index()
            )

            sns.scatterplot(
                data=model_data_mean,
                x=f"{m}_local_entropy",
                y="KL_divergence",
                ax=axes[i],
                s=100,
                color="blue",
            )

            # Add regression line
            x = model_data_mean[f"{m}_local_entropy"]
            y = model_data_mean["KL_divergence"]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            axes[i].plot(line_x, line_y, color="red", linestyle="--", alpha=0.3)

            # Add R² value
            r_squared = r_value**2
            axes[i].text(
                0.05,
                0.95,
                f"R² = {r_squared:.3f}",
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=14,
            )

            axes[i].set_xlabel(f"{m}-local Entropy", fontsize=16)
            axes[i].set_ylabel("KL Divergence" if i == 0 else "", fontsize=16)
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(f"{architecture_map[architecture]}", fontsize=20, y=0.9)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        yield fig


def plot_next_sym_global_entropy_vs_cross_entropy(df, architecture_map):
    """Plot next symbol (global) entropy vs cross entropy with regression lines for each architecture."""
    # Plot for each architecture
    for architecture in df["architecture"].unique():
        # Get data for this architecture
        model_data = df[df["architecture"] == architecture]

        # Create subplot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Average across training seeds
        model_data_mean = (
            model_data.groupby("grammar_name")
            .agg(
                {
                    "next_symbol_entropy": "mean",
                    "cross_entropy_per_token_base_2": "mean",
                }
            )
            .reset_index()
        )

        sns.scatterplot(
            data=model_data_mean,
            x="next_symbol_entropy",
            y="cross_entropy_per_token_base_2",
            ax=ax,
            s=100,
            color="blue",
        )

        # Add regression line
        x = model_data_mean["next_symbol_entropy"]
        y = model_data_mean["cross_entropy_per_token_base_2"]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.array([x.min(), x.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color="red", linestyle="--", alpha=0.3)

        # Add R² value
        r_squared = r_value**2
        ax.text(
            0.05,
            0.95,
            f"R² = {r_squared:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=14,
        )

        ax.set_xlabel("Global Entropy", fontsize=16)
        ax.set_ylabel("Cross Entropy", fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"{architecture_map[architecture]}", fontsize=20, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        yield fig


def plot_next_sym_global_entropy_vs_kl_divergence(df, architecture_map):
    """Plot next symbol (global) entropy vs KL divergence with regression lines for each architecture."""
    # Plot for each architecture
    for architecture in df["architecture"].unique():
        # Get data for this architecture
        model_data = df[df["architecture"] == architecture]

        # Create subplot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Average across training seeds
        model_data_mean = (
            model_data.groupby("grammar_name")
            .agg(
                {
                    "next_symbol_entropy": "mean",
                    "KL_divergence": "mean",
                }
            )
            .reset_index()
        )

        sns.scatterplot(
            data=model_data_mean,
            x="next_symbol_entropy",
            y="KL_divergence",
            ax=ax,
            s=100,
            color="blue",
        )

        # Add regression line
        x = model_data_mean["next_symbol_entropy"]
        y = model_data_mean["KL_divergence"]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.array([x.min(), x.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color="red", linestyle="--", alpha=0.3)

        # Add R² value
        r_squared = r_value**2
        ax.text(
            0.05,
            0.95,
            f"R² = {r_squared:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=14,
        )

        ax.set_xlabel("Global Entropy", fontsize=16)
        ax.set_ylabel("KL Divergence", fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"{architecture_map[architecture]}", fontsize=20, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        yield fig


def plot_string_global_entropy_vs_cross_entropy(df, architecture_map):
    """Plot string-level global entropy vs cross entropy with regression lines for each architecture."""
    # Plot for each architecture
    for architecture in df["architecture"].unique():
        # Get data for this architecture
        model_data = df[df["architecture"] == architecture]

        # Create subplot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Average across training seeds
        model_data_mean = (
            model_data.groupby("grammar_name")
            .agg(
                {
                    "entropy": "mean",
                    "cross_entropy_per_token_base_2": "mean",
                }
            )
            .reset_index()
        )

        sns.scatterplot(
            data=model_data_mean,
            x="entropy",
            y="cross_entropy_per_token_base_2",
            ax=ax,
            s=100,
            color="blue",
        )

        # Add regression line
        x = model_data_mean["entropy"]
        y = model_data_mean["cross_entropy_per_token_base_2"]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.array([x.min(), x.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color="red", linestyle="--", alpha=0.3)

        # Add R² value
        r_squared = r_value**2
        ax.text(
            0.05,
            0.95,
            f"R² = {r_squared:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=14,
        )

        ax.set_xlabel("String-level Global Entropy", fontsize=16)
        ax.set_ylabel("Cross Entropy", fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"{architecture_map[architecture]}", fontsize=20, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        yield fig


def plot_string_global_entropy_vs_kl_divergence(df, architecture_map):
    """Plot string-level global entropy vs KL divergence with regression lines for each architecture."""
    # Plot for each architecture
    for architecture in df["architecture"].unique():
        # Get data for this architecture
        model_data = df[df["architecture"] == architecture]

        # Create subplot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Average across training seeds
        model_data_mean = (
            model_data.groupby("grammar_name")
            .agg(
                {
                    "entropy": "mean",
                    "KL_divergence": "mean",
                }
            )
            .reset_index()
        )

        sns.scatterplot(
            data=model_data_mean,
            x="entropy",
            y="KL_divergence",
            ax=ax,
            s=100,
            color="blue",
        )

        # Add regression line
        x = model_data_mean["entropy"]
        y = model_data_mean["KL_divergence"]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.array([x.min(), x.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color="red", linestyle="--", alpha=0.3)

        # Add R² value
        r_squared = r_value**2
        ax.text(
            0.05,
            0.95,
            f"R² = {r_squared:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=14,
        )

        ax.set_xlabel("String-level Global Entropy", fontsize=16)
        ax.set_ylabel("KL Divergence", fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"{architecture_map[architecture]}", fontsize=20, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        yield fig


def parse_args():
    parser = argparse.ArgumentParser(description="Plot entropy analysis results")
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

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create architecture mapping
    architecture_map = dict(zip(args.architectures, args.architecture_labels))

    # Read data
    df = pd.read_csv(args.results_path)

    # Calculate KL divergence
    df["KL_divergence"] = (
        df["cross_entropy_per_token_base_2"] - df["next_symbol_entropy"]
    )

    # Filter for specified architectures
    df = df[df["architecture"].isin(args.architectures)]

    # Create plots
    entropy_fig = plot_entropy_distributions(df)

    # Save plots
    entropy_fig.savefig(output_dir / "entropy_distributions.png")

    # Create and save local entropy vs cross entropy plots
    for i, fig in enumerate(plot_local_entropy_vs_cross_entropy(df, architecture_map)):
        arch = list(df["architecture"].unique())[i]
        fig.savefig(
            output_dir
            / f"m_local_entropy_vs_cross_entropy_{arch}_{args.split_name}.png"
        )
        plt.close(fig)

    # Create and save local entropy vs KL divergence plots
    for i, fig in enumerate(plot_local_entropy_vs_kl_divergence(df, architecture_map)):
        arch = list(df["architecture"].unique())[i]
        fig.savefig(
            output_dir
            / f"m_local_entropy_vs_KL_divergence_{arch}_{args.split_name}.png"
        )
        plt.close(fig)

    # Create and save global entropy vs cross entropy plots
    for i, fig in enumerate(
        plot_next_sym_global_entropy_vs_cross_entropy(df, architecture_map)
    ):
        arch = list(df["architecture"].unique())[i]
        fig.savefig(
            output_dir
            / f"next_sym_global_entropy_vs_cross_entropy_{arch}_{args.split_name}.png"
        )
        plt.close(fig)

    # Create and save global entropy vs KL divergence plots
    for i, fig in enumerate(
        plot_next_sym_global_entropy_vs_kl_divergence(df, architecture_map)
    ):
        arch = list(df["architecture"].unique())[i]
        fig.savefig(
            output_dir
            / f"next_sym_global_entropy_vs_KL_divergence_{arch}_{args.split_name}.png"
        )
        plt.close(fig)

    # Create and save string-level global entropy vs cross entropy plots
    for i, fig in enumerate(
        plot_string_global_entropy_vs_cross_entropy(df, architecture_map)
    ):
        arch = list(df["architecture"].unique())[i]
        fig.savefig(
            output_dir
            / f"string_global_entropy_vs_cross_entropy_{arch}_{args.split_name}.png"
        )
        plt.close(fig)

    # Create and save string-level global entropy vs KL divergence plots
    for i, fig in enumerate(
        plot_string_global_entropy_vs_kl_divergence(df, architecture_map)
    ):
        arch = list(df["architecture"].unique())[i]
        fig.savefig(
            output_dir
            / f"string_global_entropy_vs_KL_divergence_{arch}_{args.split_name}.png"
        )
        plt.close(fig)

    plt.close("all")


if __name__ == "__main__":
    main()
