import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
import polars as pl


def calculate_entropy_and_perplexity(counts_df, total_samples):
    counts_df = counts_df.with_columns((pl.col("count") / total_samples).alias("prob"))
    counts_df = counts_df.with_columns(pl.col("prob").log().alias("log_prob"))
    counts_df = counts_df.with_columns(
        (-pl.col("prob") * pl.col("log_prob")).alias("entropy_component")
    )
    entropy = counts_df["entropy_component"].sum()
    perplexity = 2**entropy
    return entropy, perplexity


def main():
    parser = argparse.ArgumentParser(
        description="Calculate entropy and perplexity from sentence counts"
    )
    parser.add_argument(
        "--input_file",
        type=lambda p: Path(p).resolve(),
        required=True,
        help="Input file path (sentence_counts.json.gz)",
    )
    parser.add_argument(
        "--output_dir",
        type=lambda p: Path(p).resolve(),
        required=True,
        help="Output directory for entropy and perplexity values",
    )
    # sample_size 引数を追加
    parser.add_argument(
        "--sample_size",
        type=int,
        required=True,
        help="Total number of samples used for probability calculation",
    )
    parser.add_argument("--output_suffix", type=str, default="_from_samples")
    args = parser.parse_args()
    print(args)

    print(f"Reading counts from {args.input_file}...")
    if ".json.gz" in args.input_file.name:
        with gzip.open(args.input_file, "rt", encoding="utf-8") as f:
            breakpoint()
            counts = json.load(f)
        counts_df = pl.DataFrame(
            {
                "sentence": list(counts.keys()),
                "count": list(counts.values()),
            }
        )
    elif ".csv.gz" in args.input_file.name:
        counts_df = pl.read_csv(args.input_file, new_columns=["sentence", "count"])
        counts_df = counts_df.filter(pl.col("count").is_not_null())
    else:
        raise ValueError(f"Unsupported file format: {args.input_file.suffix}")

    print("Calculating entropy and perplexity...")
    # sample_size を渡すように変更
    entropy, perplexity = calculate_entropy_and_perplexity(counts_df, args.sample_size)
    breakpoint()

    # 結果の表示
    print(f"\nResults:")
    print(f"Sample size: {counts_df['count'].sum()}")
    print(f"Entropy: {entropy:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    # 結果の保存
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / f"entropy{args.output_suffix}.value", "w") as f:
        f.write(str(entropy))

    with open(args.output_dir / f"perplexity{args.output_suffix}.value", "w") as f:
        f.write(str(perplexity))

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
