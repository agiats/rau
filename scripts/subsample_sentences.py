import argparse
import gzip
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Output first N lines from samples file"
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
        help="Output file path (output.txt.gz)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of samples to output (default: 10)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate lines before sampling",
    )
    args = parser.parse_args()

    # 出力ディレクトリの作成
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # 入力ファイルを読み込んで出力ファイルに書き込み
    print(f"Reading from {args.input_file}...")

    with gzip.open(args.input_file, "rt", encoding="utf-8") as fin:
        if args.deduplicate:
            lines = list(set(fin.readlines()))
            print(f"Removed duplicates: {len(lines)} unique lines")
            lines = lines[: args.sample_size]
        else:
            lines = []
            for i, line in enumerate(fin):
                if i >= args.sample_size:
                    break
                lines.append(line)

    # 結果の書き込み
    print(f"Writing {len(lines)} lines to {args.output_file}...")
    with gzip.open(args.output_file, "wt", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line)

    print(f"Output written to {args.output_file}")


if __name__ == "__main__":
    main()
