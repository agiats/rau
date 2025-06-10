#!/usr/bin/env python3
"""
Script to compare true vs estimated mlocal entropy values for PFSA grammars.

Usage:
    python compare_mlocal_entropy.py [--output_dir OUTPUT_DIR]

The script will:
1. Find all grammar directories in data/PFSA/local_entropy/
2. Compare true mlocal entropy values (metadata.json) with estimated values (metadata_kenlm_mlocal_entropy.json)
3. Generate visualizations and summary statistics
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_entropy_data(grammar_dir: Path) -> Tuple[Dict, Dict]:
    """Load true and estimated mlocal entropy values for a given grammar directory."""
    # Load true mlocal entropy values
    with open(grammar_dir / "metadata.json", "r") as f:
        true_data = json.load(f)

    # Load estimated mlocal entropy values
    with open(grammar_dir / "metadata_kenlm_mlocal_entropy.json", "r") as f:
        estimated_data = json.load(f)

    return true_data, estimated_data


def compare_single_grammar(grammar_dir: Path) -> pd.DataFrame:
    """
    Compare true vs estimated mlocal entropy values for a single grammar.

    Args:
        grammar_dir: Path to the grammar directory

    Returns:
        DataFrame with comparison results
    """
    grammar_name = grammar_dir.name
    print(f"Analyzing grammar: {grammar_name}")

    try:
        true_data, estimated_data = load_entropy_data(grammar_dir)

        # Extract mlocal entropy values
        true_mlocal = true_data["local_entropy"]
        estimated_mlocal = estimated_data["local_entropy"]

        # Create a DataFrame for comparison
        comparison_df = pd.DataFrame({
            "m": list(true_mlocal.keys()),  # Renamed from "Context Length" to "m"
            "True mlocal Entropy": list(true_mlocal.values()),
            "Estimated mlocal Entropy": list(estimated_mlocal.values())
        })

        # Calculate absolute and relative errors
        comparison_df["Absolute Error"] = abs(comparison_df["True mlocal Entropy"] - comparison_df["Estimated mlocal Entropy"])
        comparison_df["Relative Error (%)"] = 100 * comparison_df["Absolute Error"] / comparison_df["True mlocal Entropy"]
        comparison_df["Grammar"] = grammar_name

        return comparison_df

    except Exception as e:
        print(f"Error processing {grammar_name}: {e}")
        return pd.DataFrame()


def analyze_multiple_grammars(base_dir: Path, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Analyze true vs estimated mlocal entropy values for multiple grammars.

    Args:
        base_dir: Base directory containing grammar directories
        output_dir: Directory to save visualizations (optional)

    Returns:
        DataFrame with comparison results for all grammars
    """
    # Find all grammar directories
    grammar_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    if not grammar_dirs:
        print(f"No grammar directories found in {base_dir}")
        return pd.DataFrame()

    print(f"Found {len(grammar_dirs)} grammar directories")

    # Process each grammar directory
    all_results = []
    for grammar_dir in grammar_dirs:
        results = compare_single_grammar(grammar_dir)
        if not results.empty:
            all_results.append(results)

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        # Generate visualizations if output_dir is provided
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
            generate_visualizations(combined_results, output_dir)
            generate_markdown_table(combined_results, output_dir)

        return combined_results

    return pd.DataFrame()


def generate_markdown_table(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate a markdown table showing the loss between true and estimated mlocal entropy for each m value.

    Args:
        df: DataFrame with comparison results
        output_dir: Directory to save the markdown file
    """
    # Create a pivot table to organize data by m value
    m_values = sorted(df["m"].unique())
    grammars = sorted(df["Grammar"].unique())

    # Initialize markdown content
    md_content = "# mlocal Entropy Comparison: True vs Estimated\n\n"
    md_content += "This table shows the loss between true and estimated values for each context length (m).\n\n"

    # Create a summary table with mean, std, and max error metrics
    md_content += "## Summary Across All Grammars\n\n"
    md_content += "| m | Mean Absolute Error ± Std | Mean Relative Error (%) ± Std | Max Absolute Error | Max Relative Error (%) |\n"
    md_content += "|---|---------------------------|-------------------------------|-------------------|------------------------|\n"

    for m in m_values:
        subset = df[df["m"] == m]
        abs_err_mean = subset["Absolute Error"].mean()
        abs_err_std = subset["Absolute Error"].std()
        rel_err_mean = subset["Relative Error (%)"].mean()
        rel_err_std = subset["Relative Error (%)"].std()
        abs_err_max = subset["Absolute Error"].max()
        rel_err_max = subset["Relative Error (%)"].max()

        md_content += f"| {m} | {abs_err_mean:.4f} ± {abs_err_std:.4f} | {rel_err_mean:.2f}% ± {rel_err_std:.2f}% | {abs_err_max:.4f} | {rel_err_max:.2f}% |\n"

    md_content += "\n"

    # Add detailed tables for each grammar
    if len(grammars) > 1:
        md_content += "## Details by Grammar\n\n"

        for grammar in grammars:
            md_content += f"### {grammar}\n\n"
            md_content += "| m | True | Estimated | Absolute Error | Relative Error (%) |\n"
            md_content += "|---|------|-----------|----------------|-------------------|\n"

            grammar_data = df[df["Grammar"] == grammar].sort_values("m")
            for _, row in grammar_data.iterrows():
                md_content += f"| {row['m']} | {row['True mlocal Entropy']:.4f} | {row['Estimated mlocal Entropy']:.4f} | {row['Absolute Error']:.4f} | {row['Relative Error (%)']:.2f}% |\n"

            # Add summary statistics for this grammar
            abs_err_mean = grammar_data["Absolute Error"].mean()
            rel_err_mean = grammar_data["Relative Error (%)"].mean()
            md_content += f"\n**Grammar Summary**: Mean Absolute Error = {abs_err_mean:.4f}, Mean Relative Error = {rel_err_mean:.2f}%\n\n"
    else:
        # If there's only one grammar, show its detailed table
        grammar = grammars[0]
        md_content += f"## Detailed Results for {grammar}\n\n"
        md_content += "| m | True | Estimated | Absolute Error | Relative Error (%) |\n"
        md_content += "|---|------|-----------|----------------|-------------------|\n"

        grammar_data = df[df["Grammar"] == grammar].sort_values("m")
        for _, row in grammar_data.iterrows():
            md_content += f"| {row['m']} | {row['True mlocal Entropy']:.4f} | {row['Estimated mlocal Entropy']:.4f} | {row['Absolute Error']:.4f} | {row['Relative Error (%)']:.2f}% |\n"

        md_content += "\n"

    # Add correlation analysis
    if len(m_values) > 1:
        corr = df[["m", "Absolute Error", "Relative Error (%)"]].corr()
        md_content += "## Correlation Analysis\n\n"
        md_content += f"- **Correlation between m and Absolute Error**: {corr.iloc[0, 1]:.4f}\n"
        md_content += f"- **Correlation between m and Relative Error**: {corr.iloc[0, 2]:.4f}\n\n"

        if corr.iloc[0, 2] > 0.7:
            md_content += "**Note**: Strong positive correlation between m and relative error suggests that the estimation becomes less accurate for longer contexts.\n\n"
        elif corr.iloc[0, 2] < -0.7:
            md_content += "**Note**: Strong negative correlation between m and relative error suggests that the estimation becomes more accurate for longer contexts.\n\n"

    # Add overall statistics
    md_content += "## Overall Statistics\n\n"
    md_content += f"- **Mean Absolute Error**: {df['Absolute Error'].mean():.4f} ± {df['Absolute Error'].std():.4f}\n"
    md_content += f"- **Mean Relative Error**: {df['Relative Error (%)'].mean():.2f}% ± {df['Relative Error (%)'].std():.2f}%\n"
    md_content += f"- **Median Absolute Error**: {df['Absolute Error'].median():.4f}\n"
    md_content += f"- **Median Relative Error**: {df['Relative Error (%)'].median():.2f}%\n"
    md_content += f"- **Max Absolute Error**: {df['Absolute Error'].max():.4f}\n"
    md_content += f"- **Max Relative Error**: {df['Relative Error (%)'].max():.2f}%\n\n"

    # Save the markdown file
    md_path = output_dir / "mlocal_entropy_comparison.md"
    with open(md_path, "w") as f:
        f.write(md_content)

    print(f"Markdown table saved to {md_path}")


def generate_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate visualizations for the combined results.

    Args:
        df: DataFrame with comparison results for all grammars
        output_dir: Directory to save visualizations
    """
    # Create plots for each context length (m)
    m_values = sorted(df["m"].unique())

    # 1. True vs Estimated Entropy scatter plot
    plt.figure(figsize=(12, 10))
    for m in m_values:
        subset = df[df["m"] == m]
        plt.scatter(
            subset["True mlocal Entropy"],
            subset["Estimated mlocal Entropy"],
            label=f"m = {m}",
            alpha=0.7,
            s=80
        )

    # Add diagonal line (perfect estimation)
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),
        np.max([plt.xlim()[1], plt.ylim()[1]]),
    ]
    plt.plot(lims, lims, 'k--', alpha=0.8, zorder=0, label="Perfect Estimation")

    plt.xlabel("True mlocal Entropy", fontsize=12)
    plt.ylabel("Estimated mlocal Entropy", fontsize=12)
    plt.title("True vs Estimated mlocal Entropy by Context Length (m)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "true_vs_estimated_entropy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Relative Error by Context Length (m)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="m", y="Relative Error (%)")
    plt.xlabel("Context Length (m)", fontsize=12)
    plt.ylabel("Relative Error (%)", fontsize=12)
    plt.title("Relative Error by Context Length (m)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "relative_error_by_m.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Grammar-specific comparison
    if df["Grammar"].nunique() > 1:  # Only if there's more than one grammar
        # Correlation between true and estimated entropy by grammar
        plt.figure(figsize=(12, 8))
        for grammar in df["Grammar"].unique():
            subset = df[df["Grammar"] == grammar]
            plt.scatter(
                subset["m"].astype(int),
                subset["Relative Error (%)"],
                label=grammar,
                alpha=0.8,
                s=100
            )
            # Add trend line for each grammar
            z = np.polyfit(subset["m"].astype(int), subset["Relative Error (%)"], 1)
            p = np.poly1d(z)
            plt.plot(subset["m"].astype(int), p(subset["m"].astype(int)),
                    linestyle='--', alpha=0.6)

        plt.xlabel("Context Length (m)", fontsize=12)
        plt.ylabel("Relative Error (%)", fontsize=12)
        plt.title("Relative Error Trends by Grammar", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / "error_trends_by_grammar.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Heatmap of relative errors
        plt.figure(figsize=(max(12, len(df["Grammar"].unique()) * 1.5), 8))
        heatmap_data = df.pivot(index="Grammar", columns="m", values="Relative Error (%)")
        sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", fmt=".1f")
        plt.title("Relative Error (%) by Grammar and Context Length (m)", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "relative_error_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. Estimation quality
    plt.figure(figsize=(12, 8))

    # Scatter plot for each grammar with context length (m) as color
    unique_grammars = df["Grammar"].unique()
    for i, grammar in enumerate(unique_grammars):
        grammar_df = df[df["Grammar"] == grammar]
        m_values = grammar_df["m"].astype(int)
        plt.scatter(
            grammar_df["True mlocal Entropy"],
            grammar_df["Estimated mlocal Entropy"] / grammar_df["True mlocal Entropy"],
            c=m_values,
            label=grammar,
            alpha=0.7,
            s=100,
            cmap='viridis',
            marker=f"${i+1}$"
        )

    # Add horizontal line at 1.0 (perfect estimation)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label="Perfect Estimation")

    plt.xlabel("True mlocal Entropy", fontsize=12)
    plt.ylabel("Estimated / True Ratio", fontsize=12)
    plt.title("Quality of Estimation by Grammar and Context Length (m)", fontsize=14)
    plt.colorbar(label="Context Length (m)")
    plt.grid(True, linestyle='--', alpha=0.7)
    if len(unique_grammars) < 10:  # Only show legend if not too many grammars
        plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "estimation_quality.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Absolute error comparison
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="True mlocal Entropy",
        y="Absolute Error",
        hue="m",
        style="Grammar" if df["Grammar"].nunique() < 6 else None,  # Only use style if few grammars
        s=100,
        alpha=0.7
    )
    plt.xlabel("True mlocal Entropy", fontsize=12)
    plt.ylabel("Absolute Error", fontsize=12)
    plt.title("Absolute Error vs True mlocal Entropy", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "absolute_error_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Error vs m
    plt.figure(figsize=(10, 8))
    for grammar in df["Grammar"].unique():
        subset = df[df["Grammar"] == grammar]
        plt.plot(
            subset["m"].astype(int),
            subset["Relative Error (%)"],
            marker='o',
            label=grammar,
            linewidth=2,
            markersize=8
        )

    plt.xlabel("m", fontsize=14)
    plt.ylabel("Relative Error (%)", fontsize=14)
    plt.title("Relative Error vs Context Length (m)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    if len(df["Grammar"].unique()) < 15:  # Only show legend if not too many grammars
        plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_m.png", dpi=300, bbox_inches="tight")
    plt.close()


def summarize_results(results_df: pd.DataFrame) -> None:
    """Print summary statistics for the results."""
    if results_df.empty:
        print("No results to summarize.")
        return

    print("\nSummary Statistics:")
    print("-" * 80)

    # Overall statistics
    print(f"Number of grammars analyzed: {results_df['Grammar'].nunique()}")
    print(f"Context lengths (m) analyzed: {sorted(results_df['m'].unique())}")

    # Summary by context length (m)
    print("\nError by Context Length (m):")
    for m in sorted(results_df["m"].unique()):
        subset = results_df[results_df["m"] == m]
        mean_abs_err = subset["Absolute Error"].mean()
        mean_rel_err = subset["Relative Error (%)"].mean()
        print(f"  m = {m}:")
        print(f"    Mean Absolute Error: {mean_abs_err:.4f}")
        print(f"    Mean Relative Error: {mean_rel_err:.2f}%")

    # Overall error metrics
    print("\nOverall Error Metrics:")
    print(f"  Mean Absolute Error: {results_df['Absolute Error'].mean():.4f}")
    print(f"  Mean Relative Error: {results_df['Relative Error (%)'].mean():.2f}%")
    print(f"  Max Relative Error: {results_df['Relative Error (%)'].max():.2f}%")

    # Correlation analysis
    corr = results_df[["m", "Absolute Error", "Relative Error (%)"]].corr()
    print("\nCorrelation between Context Length (m) and Errors:")
    print(f"  m vs Absolute Error: {corr.iloc[0, 1]:.4f}")
    print(f"  m vs Relative Error: {corr.iloc[0, 2]:.4f}")

    # Grammar-specific summaries if multiple grammars
    if results_df["Grammar"].nunique() > 1:
        print("\nError by Grammar:")
        for grammar in sorted(results_df["Grammar"].unique()):
            subset = results_df[results_df["Grammar"] == grammar]
            mean_abs_err = subset["Absolute Error"].mean()
            mean_rel_err = subset["Relative Error (%)"].mean()
            print(f"  {grammar}:")
            print(f"    Mean Absolute Error: {mean_abs_err:.4f}")
            print(f"    Mean Relative Error: {mean_rel_err:.2f}%")

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare true vs estimated mlocal entropy values for PFSA grammars")
    parser.add_argument("--exp_name", type=str, default="local_entropy_rebuttal_small",
                        help="Experiment name (default: local_entropy_rebuttal_small)")
    parser.add_argument("--output_dir", type=str, default="results/figures/mlocal_entropy",
                        help="Directory to save visualizations (default: figures/mlocal_entropy)")
    args = parser.parse_args()

    base_dir = Path(f"data/PFSA/{args.exp_name}")
    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"Looking for grammar directories in {base_dir}")

    # Analyze all grammars
    results = analyze_multiple_grammars(base_dir, output_dir)

    # Summarize the results
    summarize_results(results)

    # Save results to CSV if output_dir is provided
    if output_dir is not None and not results.empty:
        output_dir.mkdir(exist_ok=True, parents=True)
        results.to_csv(output_dir / "mlocal_entropy_comparison.csv", index=False)
        print(f"Results saved to {output_dir / 'mlocal_entropy_comparison.csv'}")


if __name__ == "__main__":
    main()
