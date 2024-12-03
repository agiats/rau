import polars as pl
from pathlib import Path
import argparse
import gzip
import json


def sample_sentences(
    input_path: str,
    target_samples_per_count: int,
    output_path: str | None = None,
) -> list[str]:
    """
    Perform count-balanced sampling of sentences from a gzipped text file.

    This function:
    1. Reads sentences from gzipped file and counts their occurrences
    2. For each count group, samples up to target_samples_per_count sentences
    3. If a group has fewer samples than target_samples_per_count, takes all available samples

    Args:
        input_path: Path to input gzipped text file
        target_samples_per_count: Target number of samples to take from each count group
        output_path: Optional path to save sampled sentences (will be gzipped)

    Returns:
        List of sampled sentences
    """
    # Read sentence counts from gzipped csv
    df = pl.read_csv(input_path, new_columns=["sentence", "count"]).filter(
        pl.col("count").is_not_null()
    )

    # Print count distribution before sampling
    count_dist = df.group_by("count").agg(pl.count().alias("frequency")).sort("count")
    print("\nCount distribution before sampling:")
    print(count_dist)

    # Sample from each count group
    balanced_df = (
        df.group_by("count")
        .agg(
            pl.col("sentence").sample(
                n=pl.when(pl.count() >= target_samples_per_count)
                .then(target_samples_per_count)
                .otherwise(pl.count())
            )
        )
        .sort("count")
    )

    sampled_dist = (
        balanced_df.explode("sentence")
        .group_by("count")
        .agg(pl.count("sentence").alias("sampled"))
        .sort("count")
    )
    print("\nSamples obtained from each count group:")
    print(sampled_dist)

    # Convert results to a flat list of sentences
    sampled_sentences = balanced_df.get_column("sentence").explode().to_list()

    # filter
    df = df.filter(
        pl.col("sentence").is_in(sampled_sentences) & pl.col("count").is_not_null()
    )

    # Optionally save results to gzipped file
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            df.write_csv(f, include_header=True)

    return sampled_sentences


def main():
    parser = argparse.ArgumentParser(
        description="Perform count-balanced sampling of sentences from a gzipped text file"
    )
    parser.add_argument(
        "--input_path",
        help="Path to input gzipped file containing sentences (one per line)",
    )
    parser.add_argument(
        "--target-samples-per-count",
        type=int,
        default=100,
        help="Target number of samples to take from each count group (default: 100)",
    )
    parser.add_argument(
        "--output-path", help="Path to save sampled sentences (will be gzipped)"
    )

    args = parser.parse_args()

    # Execute sampling
    sampled_sentences = sample_sentences(
        args.input_path, args.target_samples_per_count, args.output_path
    )

    # Print final summary
    print(f"\nTotal sentences sampled: {len(sampled_sentences)}")


if __name__ == "__main__":
    main()
