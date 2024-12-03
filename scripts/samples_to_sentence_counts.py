import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from tqdm import tqdm
import polars as pl


def main():
    parser = argparse.ArgumentParser(
        description="Convert samples.txt.gz to sentence counts"
    )
    parser.add_argument(
        "--input_file",
        type=lambda p: Path(p).resolve(),
        required=True,
        help="Input file path (samples.txt.gz)",
    )
    parser.add_argument(
        "--output_file",
        type=lambda p: Path(p).resolve(),
        required=True,
        help="Output file path (sentence_counts.json.gz)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Number of samples to read (-1 for all samples)",
    )
    args = parser.parse_args()

    # カウンターの初期化
    counter = Counter()

    # 入力ファイルを読み込んでカウント
    print(f"Reading from {args.input_file}...")
    with gzip.open(args.input_file, "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if 0 <= args.sample_size == i:
                break
            sentence = line.strip()
            counter[sentence] += 1

    # 統計情報の表示
    total_samples = sum(counter.values())
    unique_samples = len(counter)
    print(f"\nStatistics:")
    print(f"Total samples: {total_samples}")
    print(f"Unique samples: {unique_samples} ({unique_samples/total_samples*100:.2f}%)")
    print(f"Min count: {min(counter.values())}")
    print(f"Max count: {max(counter.values())}")

    # 出力ファイルの保存
    print(f"\nWriting counts to {args.output_file}...")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "sentence": list(counter.keys()),
            "count": list(counter.values()),
        }
    )
    with gzip.open(args.output_file, "wt", encoding="utf-8") as f:
        df.write_csv(f, include_header=True)


if __name__ == "__main__":
    main()
