#!/usr/bin/env python3
"""
Generate a report summarizing the mlocal entropy comparison results.

Usage:
    python generate_mlocal_entropy_report.py [--input_csv INPUT_CSV] [--output_dir OUTPUT_DIR]

This script takes the CSV output from compare_mlocal_entropy.py and generates
a detailed HTML report with findings and visualizations.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import jinja2
import json


def load_data(input_csv: Path) -> pd.DataFrame:
    """Load data from the comparison CSV file."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file {input_csv} not found.")

    return pd.DataFrame(pd.read_csv(input_csv))


def generate_trend_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze trends in the data.

    Returns:
        Dictionary with trend analysis
    """
    analysis = {
        "total_grammars": df["Grammar"].nunique(),
        "m_values": sorted(df["m"].unique()),
        "mean_error_by_m": {},
        "correlation": {},
        "findings": []
    }

    # Calculate mean error by context length
    for m in analysis["m_values"]:
        subset = df[df["m"] == m]
        analysis["mean_error_by_m"][m] = {
            "absolute": subset["Absolute Error"].mean(),
            "relative": subset["Relative Error (%)"].mean()
        }

    # Calculate correlations
    corr = df[["m", "Absolute Error", "Relative Error (%)"]].corr()
    analysis["correlation"]["m_vs_absolute"] = corr.iloc[0, 1]
    analysis["correlation"]["m_vs_relative"] = corr.iloc[0, 2]

    # Generate findings
    if analysis["correlation"]["m_vs_relative"] > 0.7:
        analysis["findings"].append({
            "title": "Strong positive correlation between m and relative error",
            "description": "As context length (m) increases, the relative error between true and estimated mlocal entropy also increases significantly. This suggests that the estimation method may be increasingly less accurate for longer contexts."
        })
    elif analysis["correlation"]["m_vs_relative"] < -0.7:
        analysis["findings"].append({
            "title": "Strong negative correlation between m and relative error",
            "description": "As context length (m) increases, the relative error between true and estimated mlocal entropy decreases significantly. This suggests that the estimation method may be increasingly more accurate for longer contexts."
        })

    # Check if estimation consistently overestimates or underestimates
    is_consistently_under = True
    is_consistently_over = True

    for _, row in df.iterrows():
        if row["True mlocal Entropy"] <= row["Estimated mlocal Entropy"]:
            is_consistently_under = False
        if row["True mlocal Entropy"] >= row["Estimated mlocal Entropy"]:
            is_consistently_over = False

    if is_consistently_under:
        analysis["findings"].append({
            "title": "Consistent underestimation",
            "description": "The estimated mlocal entropy values are consistently lower than the true values across all m values and grammars."
        })
    elif is_consistently_over:
        analysis["findings"].append({
            "title": "Consistent overestimation",
            "description": "The estimated mlocal entropy values are consistently higher than the true values across all m values and grammars."
        })

    # Check for widening or narrowing gap
    m_values = sorted(analysis["m_values"])
    if len(m_values) > 1:
        first_m = m_values[0]
        last_m = m_values[-1]
        first_rel_err = analysis["mean_error_by_m"][first_m]["relative"]
        last_rel_err = analysis["mean_error_by_m"][last_m]["relative"]

        if last_rel_err > first_rel_err * 1.5:
            analysis["findings"].append({
                "title": "Widening error gap with increasing m",
                "description": f"The relative error increases substantially from {first_rel_err:.2f}% at m = {first_m} to {last_rel_err:.2f}% at m = {last_m}. This suggests that the estimation method has more difficulty with longer contexts."
            })
        elif first_rel_err > last_rel_err * 1.5:
            analysis["findings"].append({
                "title": "Narrowing error gap with increasing m",
                "description": f"The relative error decreases substantially from {first_rel_err:.2f}% at m = {first_m} to {last_rel_err:.2f}% at m = {last_m}. This suggests that the estimation method performs better with longer contexts."
            })

    return analysis


def generate_visualizations(df: pd.DataFrame, output_dir: Path) -> list:
    """
    Generate additional visualizations for the report.

    Returns:
        List of paths to generated visualizations
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    viz_paths = []

    # 1. Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x="True mlocal Entropy", y="Estimated mlocal Entropy",
                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})

    # Add diagonal line (perfect estimation)
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),
        np.max([plt.xlim()[1], plt.ylim()[1]]),
    ]
    plt.plot(lims, lims, 'k--', alpha=0.8, zorder=0, label="Perfect Estimation")

    plt.xlabel("True mlocal Entropy")
    plt.ylabel("Estimated mlocal Entropy")
    plt.title("True vs Estimated mlocal Entropy with Regression Line")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    scatter_path = output_dir / "scatter_with_regression.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    viz_paths.append(str(scatter_path.relative_to(output_dir.parent)))

    # 2. Error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Relative Error (%)", kde=True, bins=15)
    plt.axvline(df["Relative Error (%)"].mean(), color='r', linestyle='--',
                label=f'Mean: {df["Relative Error (%)"].mean():.2f}%')
    plt.axvline(df["Relative Error (%)"].median(), color='g', linestyle='--',
                label=f'Median: {df["Relative Error (%)"].median():.2f}%')

    plt.xlabel("Relative Error (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Relative Errors")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    hist_path = output_dir / "error_distribution.png"
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    viz_paths.append(str(hist_path.relative_to(output_dir.parent)))

    # 3. Error breakdown by m
    if df["Grammar"].nunique() > 1:
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=df, x="m", y="Relative Error (%)", hue="Grammar")
        plt.xlabel("m")
        plt.ylabel("Relative Error (%)")
        plt.title("Distribution of Relative Errors by m and Grammar")
        plt.grid(True, linestyle='--', alpha=0.7, axis="y")

        violin_path = output_dir / "error_violin_by_m.png"
        plt.savefig(violin_path, dpi=300, bbox_inches="tight")
        plt.close()
        viz_paths.append(str(violin_path.relative_to(output_dir.parent)))

    return viz_paths


def generate_html_report(df: pd.DataFrame, analysis: dict, viz_paths: list, output_dir: Path) -> Path:
    """
    Generate an HTML report with the analysis results.

    Args:
        df: DataFrame with comparison results
        analysis: Dictionary with trend analysis
        viz_paths: List of paths to visualizations
        output_dir: Directory to save the report

    Returns:
        Path to the generated HTML report
    """
    # Get Jinja2 template
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>mlocal Entropy Comparison Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .summary {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .finding {
                background-color: #e9f7ef;
                padding: 15px;
                border-left: 5px solid #27ae60;
                margin-bottom: 15px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .viz-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .viz-item {
                flex: 0 0 48%;
                margin-bottom: 20px;
            }
            .viz-item img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }
            .correlation {
                display: flex;
                justify-content: space-between;
            }
            .correlation-item {
                flex: 0 0 48%;
                padding: 15px;
                background-color: #f1f8ff;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>mlocal Entropy Comparison Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p>This report compares true mlocal entropy values from metadata.json with estimated values from metadata_kenlm_mlocal_entropy.json.</p>
                <ul>
                    <li><strong>Number of grammar patterns analyzed:</strong> {{ analysis.total_grammars }}</li>
                    <li><strong>Context lengths (m) analyzed:</strong> {{ analysis.m_values|join(', ') }}</li>
                    <li><strong>Overall mean absolute error:</strong> {{ overall_stats.mean_abs_err|round(4) }}</li>
                    <li><strong>Overall mean relative error:</strong> {{ overall_stats.mean_rel_err|round(2) }}%</li>
                </ul>
            </div>

            <h2>Key Findings</h2>
            {% if analysis.findings %}
                {% for finding in analysis.findings %}
                    <div class="finding">
                        <h3>{{ finding.title }}</h3>
                        <p>{{ finding.description }}</p>
                    </div>
                {% endfor %}
            {% else %}
                <p>No significant patterns were found in the data.</p>
            {% endif %}

            <h2>Correlation Analysis</h2>
            <div class="correlation">
                <div class="correlation-item">
                    <h3>m vs Absolute Error</h3>
                    <p><strong>Correlation:</strong> {{ analysis.correlation.m_vs_absolute|round(4) }}</p>
                    <p>
                        {% if analysis.correlation.m_vs_absolute > 0.7 %}
                            Strong positive correlation: As m increases, absolute error increases.
                        {% elif analysis.correlation.m_vs_absolute < -0.7 %}
                            Strong negative correlation: As m increases, absolute error decreases.
                        {% elif analysis.correlation.m_vs_absolute > 0.3 %}
                            Moderate positive correlation: As m increases, absolute error tends to increase.
                        {% elif analysis.correlation.m_vs_absolute < -0.3 %}
                            Moderate negative correlation: As m increases, absolute error tends to decrease.
                        {% else %}
                            Weak or no correlation between m and absolute error.
                        {% endif %}
                    </p>
                </div>
                <div class="correlation-item">
                    <h3>m vs Relative Error</h3>
                    <p><strong>Correlation:</strong> {{ analysis.correlation.m_vs_relative|round(4) }}</p>
                    <p>
                        {% if analysis.correlation.m_vs_relative > 0.7 %}
                            Strong positive correlation: As m increases, relative error increases.
                        {% elif analysis.correlation.m_vs_relative < -0.7 %}
                            Strong negative correlation: As m increases, relative error decreases.
                        {% elif analysis.correlation.m_vs_relative > 0.3 %}
                            Moderate positive correlation: As m increases, relative error tends to increase.
                        {% elif analysis.correlation.m_vs_relative < -0.3 %}
                            Moderate negative correlation: As m increases, relative error tends to decrease.
                        {% else %}
                            Weak or no correlation between m and relative error.
                        {% endif %}
                    </p>
                </div>
            </div>

            <h2>Error by m</h2>
            <table>
                <tr>
                    <th>m</th>
                    <th>Mean Absolute Error</th>
                    <th>Mean Relative Error (%)</th>
                </tr>
                {% for m_val in analysis.m_values %}
                    <tr>
                        <td>{{ m_val }}</td>
                        <td>{{ analysis.mean_error_by_m[m_val].absolute|round(4) }}</td>
                        <td>{{ analysis.mean_error_by_m[m_val].relative|round(2) }}%</td>
                    </tr>
                {% endfor %}
            </table>

            <h2>Data Overview</h2>
            <table>
                <tr>
                    <th>Grammar</th>
                    <th>m</th>
                    <th>True mlocal Entropy</th>
                    <th>Estimated mlocal Entropy</th>
                    <th>Absolute Error</th>
                    <th>Relative Error (%)</th>
                </tr>
                {% for _, row in df.iterrows() %}
                    <tr>
                        <td>{{ row['Grammar'] }}</td>
                        <td>{{ row['m'] }}</td>
                        <td>{{ row['True mlocal Entropy']|round(4) }}</td>
                        <td>{{ row['Estimated mlocal Entropy']|round(4) }}</td>
                        <td>{{ row['Absolute Error']|round(4) }}</td>
                        <td>{{ row['Relative Error (%)']|round(2) }}</td>
                    </tr>
                {% endfor %}
            </table>

            <h2>Visualizations</h2>
            <div class="viz-container">
                {% for viz_path in viz_paths %}
                    <div class="viz-item">
                        <img src="../{{ viz_path }}" alt="Visualization">
                    </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """

    # Prepare template variables
    template_vars = {
        "df": df,
        "analysis": analysis,
        "viz_paths": viz_paths,
        "overall_stats": {
            "mean_abs_err": df["Absolute Error"].mean(),
            "mean_rel_err": df["Relative Error (%)"].mean()
        }
    }

    # Render template
    template = jinja2.Template(template_str)
    report_html = template.render(**template_vars)

    # Save report
    report_path = output_dir / "mlocal_entropy_report.html"
    with open(report_path, "w") as f:
        f.write(report_html)

    # Also save analysis as JSON for potential reuse
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=4)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate a report summarizing mlocal entropy comparison results")
    parser.add_argument("--input_csv", type=str, default="results/figures/mlocal_entropy/mlocal_entropy_comparison.csv",
                        help="Path to the CSV file with comparison results")
    parser.add_argument("--output_dir", type=str, default="results/figures/mlocal_entropy/report",
                        help="Directory to save the report")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Load data
        print(f"Loading data from {input_csv}")
        df = load_data(input_csv)

        # Generate trend analysis
        print("Analyzing trends in the data")
        analysis = generate_trend_analysis(df)

        # Generate visualizations
        print("Generating visualizations")
        viz_paths = generate_visualizations(df, output_dir)

        # Generate HTML report
        print("Generating HTML report")
        report_path = generate_html_report(df, analysis, viz_paths, output_dir)

        print(f"Report successfully generated: {report_path}")

    except Exception as e:
        print(f"Error generating report: {e}")


if __name__ == "__main__":
    main()
