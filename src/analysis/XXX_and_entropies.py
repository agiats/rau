#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def read_metadata_files(data_dir):
    """
    Read all metadata.json files in the data directory.

    Args:
        data_dir (str or Path): Path to the directory containing metadata.json files

    Returns:
        list: List of dictionaries containing metadata information
    """
    metadata_list = []
    data_path = Path(data_dir)

    # Find all metadata.json files using pathlib
    for metadata_path in data_path.rglob("metadata.json"):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                # Add the file path for reference
                metadata["file_path"] = str(metadata_path)
                metadata_list.append(metadata)
        except json.JSONDecodeError:
            print(f"Error parsing {metadata_path}, skipping file")
        except Exception as e:
            print(f"Error reading {metadata_path}: {e}")

    print(f"Found {len(metadata_list)} metadata files")
    return metadata_list


def create_dataframe(metadata_list):
    """
    Create a pandas DataFrame from metadata list.

    Args:
        metadata_list (list): List of metadata dictionaries

    Returns:
        DataFrame: Pandas DataFrame with metadata information
    """
    # Initialize lists to store data
    all_data = []

    for metadata in metadata_list:
        # Get XXX value
        xxx_value = metadata.get("XXX")

        # Extract local entropy values
        local_entropy_dict = metadata.get("local_entropy", {})

        # Extract prefix local entropy values
        prefix_local_entropy_dict = metadata.get("prefix_local_entropy", {})

        # Create a row for each context length
        for context_length in set(list(local_entropy_dict.keys()) + list(prefix_local_entropy_dict.keys())):
            row = {
                "file_path": metadata.get("file_path"),
                "n_states": metadata.get("n_states"),
                "N_sym": metadata.get("N_sym"),
                "topology_seed": metadata.get("topology_seed"),
                "weight_seed": metadata.get("weight_seed"),
                "XXX": xxx_value,
                "context_length": context_length,
                "local_entropy": local_entropy_dict.get(context_length),
                "prefix_local_entropy": prefix_local_entropy_dict.get(context_length)
            }
            all_data.append(row)

    return pd.DataFrame(all_data)


def plot_grouped_by_entropy_type(df, results_dir):
    """
    Create plots grouped by entropy type, with subplots for each context length.

    Args:
        df (DataFrame): DataFrame containing metadata information
        results_dir (Path): Path to save plots
    """
    # Get sorted context lengths
    context_lengths = sorted(df['context_length'].unique())

    if not context_lengths:
        print("No context lengths found in data")
        return

    # Create a 2x2 grid of subplots for each entropy type
    fig_local, axes_local = plt.subplots(2, 2, figsize=(16, 12))
    fig_prefix, axes_prefix = plt.subplots(2, 2, figsize=(16, 12))

    # Flatten axes arrays for easier indexing
    axes_local = axes_local.flatten()
    axes_prefix = axes_prefix.flatten()

    # For each context length, create a subplot
    for i, context_length in enumerate(context_lengths):
        if i >= len(axes_local):  # Skip if we have more context lengths than subplot positions
            print(f"Warning: More context lengths than subplot positions. Skipping context length {context_length}")
            continue

        # Get data for this context length
        context_df = df[df['context_length'] == context_length].copy()

        # Plot local entropy vs XXX
        local_df = context_df.dropna(subset=['local_entropy', 'XXX'])
        if not local_df.empty:
            ax = axes_local[i]

            # Create scatter plot
            sns.scatterplot(
                data=local_df,
                x='local_entropy',
                y='XXX',
                ax=ax,
                s=100,
                alpha=0.7
            )

            # Add regression line
            x = local_df['local_entropy']
            y = local_df['XXX']
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, color='red', linestyle='--', alpha=0.7)

            # Add R² value
            r_squared = r_value**2
            ax.text(
                0.05, 0.95,
                f"R² = {r_squared:.3f}\np = {p_value:.3e}",
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7)
            )

            # Set labels and title
            ax.set_xlabel(f"{context_length}-gram Local Entropy", fontsize=12)
            ax.set_ylabel("XXX", fontsize=12)
            ax.set_title(f"{context_length}-gram Context", fontsize=14)
            ax.grid(True, alpha=0.3)

        # Plot prefix local entropy vs XXX
        prefix_df = context_df.dropna(subset=['prefix_local_entropy', 'XXX'])
        if not prefix_df.empty:
            ax = axes_prefix[i]

            # Create scatter plot
            sns.scatterplot(
                data=prefix_df,
                x='prefix_local_entropy',
                y='XXX',
                ax=ax,
                s=100,
                alpha=0.7
            )

            # Add regression line
            x = prefix_df['prefix_local_entropy']
            y = prefix_df['XXX']
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, color='red', linestyle='--', alpha=0.7)

            # Add R² value
            r_squared = r_value**2
            ax.text(
                0.05, 0.95,
                f"R² = {r_squared:.3f}\np = {p_value:.3e}",
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7)
            )

            # Set labels and title
            ax.set_xlabel(f"{context_length}-gram Prefix Local Entropy", fontsize=12)
            ax.set_ylabel("XXX", fontsize=12)
            ax.set_title(f"{context_length}-gram Context", fontsize=14)
            ax.grid(True, alpha=0.3)

    # Set overall titles
    fig_local.suptitle("Relationship between Local Entropy and XXX across Context Lengths", fontsize=16, y=0.98)
    fig_prefix.suptitle("Relationship between Prefix Local Entropy and XXX across Context Lengths", fontsize=16, y=0.98)

    # Adjust layout and save
    fig_local.tight_layout(rect=[0, 0, 1, 0.96])
    fig_prefix.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figures
    local_output_path = results_dir / "local_entropy_vs_XXX.png"
    prefix_output_path = results_dir / "prefix_local_entropy_vs_XXX.png"

    fig_local.savefig(local_output_path, dpi=300)
    fig_prefix.savefig(prefix_output_path, dpi=300)

    plt.close(fig_local)
    plt.close(fig_prefix)

    print(f"Saved grouped plots to {results_dir}")


def plot_grouped_by_states_and_symbols(df, results_dir):
    """
    Create plots with points grouped by number of states and symbols.
    States are represented by different marker shapes, and
    symbols are represented by different colors.

    Args:
        df (DataFrame): DataFrame containing metadata information
        results_dir (Path): Path to save plots
    """
    # Get unique values for n_states and N_sym
    n_states_values = sorted(df['n_states'].dropna().unique())
    n_sym_values = sorted(df['N_sym'].dropna().unique())

    # Create marker and color mappings
    # Define markers for n_states
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    n_states_markers = {n: markers[i % len(markers)] for i, n in enumerate(n_states_values)}

    # Define colors for N_sym using default color cycle
    colors = plt.cm.tab10.colors
    n_sym_colors = {n: colors[i % len(colors)] for i, n in enumerate(n_sym_values)}

    # Get sorted context lengths
    context_lengths = sorted(df['context_length'].unique())

    if not context_lengths:
        print("No context lengths found in data")
        return

    # Create figures for local and prefix local entropy
    for entropy_type in ['local_entropy', 'prefix_local_entropy']:
        # Create a 2x2 grid (or adjust based on context_lengths)
        rows = (len(context_lengths) + 1) // 2
        cols = min(2, len(context_lengths))

        # Main figure size (larger to accommodate legend at bottom)
        fig = plt.figure(figsize=(12, 8 + 2))  # Extra space for legend

        # Create grid with space for the legend at the bottom
        gs = fig.add_gridspec(rows + 1, cols, height_ratios=list([1] * rows) + [0.2])

        # Create subplots
        axes = []
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(context_lengths):
                    axes.append(fig.add_subplot(gs[i, j]))

        # For each context length, create a subplot
        for i, context_length in enumerate(context_lengths):
            if i >= len(axes):
                print(f"Warning: More context lengths than subplot positions. Skipping context length {context_length}")
                continue

            ax = axes[i]

            # Get data for this context length
            context_df = df[df['context_length'] == context_length].copy()

            # Filter for this entropy type
            entropy_df = context_df.dropna(subset=[entropy_type, 'XXX'])

            if entropy_df.empty:
                continue

            # Group by n_states and N_sym
            for n_states in n_states_values:
                for n_sym in n_sym_values:
                    # Filter data for this combination
                    subset = entropy_df[(entropy_df['n_states'] == n_states) &
                                       (entropy_df['N_sym'] == n_sym)]

                    if not subset.empty:
                        # Plot points
                        ax.scatter(
                            subset[entropy_type],
                            subset['XXX'],
                            marker=n_states_markers[n_states],
                            color=n_sym_colors[n_sym],
                            s=100,
                            alpha=0.7,
                            label=f'|Q|={n_states}, |Σ|={n_sym}'
                        )

                        # Add regression line if we have enough points
                        if len(subset) > 2:
                            x = subset[entropy_type]
                            y = subset['XXX']
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            line_x = np.array([x.min(), x.max()])
                            line_y = slope * line_x + intercept
                            ax.plot(
                                line_x,
                                line_y,
                                color=n_sym_colors[n_sym],
                                linestyle='--',
                                alpha=0.5
                            )

            # Set labels and title
            entropy_label = "Local Entropy" if entropy_type == 'local_entropy' else "Prefix Local Entropy"
            ax.set_xlabel(f"{context_length}-gram {entropy_label}", fontsize=12)
            ax.set_ylabel("XXX", fontsize=12)
            ax.set_title(f"{context_length}-gram Context", fontsize=14)
            ax.grid(True, alpha=0.3)

        # Add a legend at the bottom
        legend_ax = fig.add_subplot(gs[-1, :])
        legend_ax.axis('off')

        # Create legend elements for states
        state_markers = [plt.Line2D(
            [0], [0],
            marker=marker,
            color='gray',
            markerfacecolor='gray',
            markersize=10,
            linestyle='None',
            label=f'|Q|={n_states}'
        ) for n_states, marker in n_states_markers.items()]

        # Create legend elements for symbols
        sym_markers = [plt.Line2D(
            [0], [0],
            marker='o',
            color=color,
            markersize=10,
            linestyle='None',
            label=f'|Σ|={n_sym}'
        ) for n_sym, color in n_sym_colors.items()]

        # Add legends side by side
        legend1 = legend_ax.legend(
            handles=state_markers,
            title='Number of States',
            loc='center',
            bbox_to_anchor=(0.3, 0.5),
            fontsize=10,
            title_fontsize=12,
            ncol=min(len(n_states_markers), 4)
        )
        legend_ax.add_artist(legend1)

        legend2 = legend_ax.legend(
            handles=sym_markers,
            title='Number of Symbols',
            loc='center',
            bbox_to_anchor=(0.7, 0.5),
            fontsize=10,
            title_fontsize=12,
            ncol=min(len(n_sym_colors), 4)
        )

        # Set overall title
        entropy_title = "Local Entropy" if entropy_type == 'local_entropy' else "Prefix Local Entropy"
        fig.suptitle(f"Relationship between {entropy_title} and XXX by States and Symbols", fontsize=16, y=0.98)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        output_path = results_dir / f"{entropy_type}_by_states_symbols_vs_XXX.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"Saved {entropy_type} plot by states and symbols to {results_dir}")


def plot_fixed_states(df, results_dir):
    """
    Create plots where each plot shows data for a fixed number of states,
    with different colors for different number of symbols.

    Args:
        df (DataFrame): DataFrame containing metadata information
        results_dir (Path): Path to save plots
    """
    # Get unique values for n_states and N_sym
    n_states_values = sorted(df['n_states'].dropna().unique())
    n_sym_values = sorted(df['N_sym'].dropna().unique())

    # Define colors for N_sym using default color cycle
    colors = plt.cm.tab10.colors
    n_sym_colors = {n: colors[i % len(colors)] for i, n in enumerate(n_sym_values)}

    # Get sorted context lengths
    context_lengths = sorted(df['context_length'].unique())

    if not context_lengths or not n_states_values:
        print("No context lengths or states found in data")
        return

    # Create plots for each n_states value and entropy type
    for n_states in n_states_values:
        # Filter data for this number of states
        states_df = df[df['n_states'] == n_states].copy()

        if states_df.empty:
            continue

        # Plot for each entropy type
        for entropy_type in ['local_entropy', 'prefix_local_entropy']:
            # Create a 2x2 grid (or adjust based on context_lengths)
            rows = (len(context_lengths) + 1) // 2
            cols = min(2, len(context_lengths))

            fig, axes = plt.subplots(rows, cols, figsize=(12, 10))

            # Make axes a 2D array for consistent indexing
            if len(context_lengths) == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = np.array([axes])

            # For each context length, create a subplot
            for i, context_length in enumerate(context_lengths):
                # Calculate row and column for this subplot
                row_idx = i // cols
                col_idx = i % cols

                # Skip if we're out of subplot positions
                if row_idx >= rows or col_idx >= cols:
                    print(f"Warning: More context lengths than subplot positions. Skipping context length {context_length}")
                    continue

                ax = axes[row_idx, col_idx]

                # Get data for this context length
                context_df = states_df[states_df['context_length'] == context_length].copy()

                # Filter for this entropy type
                entropy_df = context_df.dropna(subset=[entropy_type, 'XXX'])

                if entropy_df.empty:
                    ax.set_visible(False)
                    continue

                # Plot for each number of symbols
                for n_sym in n_sym_values:
                    # Filter data for this combination
                    subset = entropy_df[entropy_df['N_sym'] == n_sym]

                    if not subset.empty:
                        # Plot points
                        ax.scatter(
                            subset[entropy_type],
                            subset['XXX'],
                            color=n_sym_colors[n_sym],
                            s=100,
                            alpha=0.7,
                            label=f'|Σ|={n_sym}'
                        )

                        # Add regression line if we have enough points
                        if len(subset) > 2:
                            x = subset[entropy_type]
                            y = subset['XXX']
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                            # Get a wider x-range for the regression line
                            x_range = x.max() - x.min()
                            if x_range > 0:
                                padding = 0.1 * x_range
                                line_x = np.array([x.min() - padding, x.max() + padding])
                                line_y = slope * line_x + intercept
                                ax.plot(
                                    line_x,
                                    line_y,
                                    color=n_sym_colors[n_sym],
                                    linestyle='--',
                                    alpha=0.7,
                                    linewidth=2
                                )

                                # Add R² value near the line
                                r_squared = r_value**2
                                ax.text(
                                    x.mean(),
                                    intercept + slope * x.mean(),
                                    f"R² = {r_squared:.3f}",
                                    color=n_sym_colors[n_sym],
                                    fontsize=10,
                                    ha='center',
                                    va='bottom',
                                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
                                )

                # Set labels and title
                entropy_label = "Local Entropy" if entropy_type == 'local_entropy' else "Prefix Local Entropy"
                ax.set_xlabel(f"{context_length}-gram {entropy_label}", fontsize=12)
                ax.set_ylabel("XXX", fontsize=12)
                ax.set_title(f"{context_length}-gram Context", fontsize=14)
                ax.grid(True, alpha=0.3)

                # Add legend for first subplot only
                if i == 0:
                    ax.legend(title="Number of Symbols", fontsize=10, title_fontsize=12)

            # Remove empty subplots if any
            for i in range(len(context_lengths), rows * cols):
                row_idx = i // cols
                col_idx = i % cols
                if row_idx < rows and col_idx < cols:
                    axes[row_idx, col_idx].set_visible(False)

            # Set overall title
            entropy_title = "Local Entropy" if entropy_type == 'local_entropy' else "Prefix Local Entropy"
            fig.suptitle(f"Relationship between {entropy_title} and XXX for Fixed States (|Q|={n_states})",
                        fontsize=16, y=0.98)

            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save figure
            output_path = results_dir / f"{entropy_type}_fixed_states_{n_states}_vs_XXX.png"
            fig.savefig(output_path, dpi=300)
            plt.close(fig)

            print(f"Saved {entropy_type} plot for fixed states (|Q|={n_states}) to {results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze relationship between local entropy measures and XXX")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing metadata.json files"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory to save plot results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create results directory if it doesn't exist
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Read all metadata files
    metadata_list = read_metadata_files(args.data_dir)

    if not metadata_list:
        print("No metadata files found. Exiting.")
        return

    # Create DataFrame from metadata
    df = create_dataframe(metadata_list)

    # Save DataFrame for reference
    df.to_csv(results_dir / "metadata_analysis.csv", index=False)

    # Generate plots grouped by entropy type
    plot_grouped_by_entropy_type(df, results_dir)

    # Generate plots grouped by states and symbols
    plot_grouped_by_states_and_symbols(df, results_dir)

    # Generate plots with fixed number of states
    plot_fixed_states(df, results_dir)

    print(f"Analysis complete. Results saved to {results_dir}")


if __name__ == "__main__":
    main()
