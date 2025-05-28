import json
import math
import sys
from pathlib import Path


def split_config(config_file: str, output_dir: str, n_splits: int):
    # 設定ファイルを読み込む
    with open(Path(config_file).resolve(), "r") as f:
        config = json.load(f)

    # 設定を分割
    perturbations = list(config.items())
    split_size = math.ceil(len(perturbations) / n_splits)
    splits = [
        dict(perturbations[i : i + split_size])
        for i in range(0, len(perturbations), split_size)
    ]

    # 出力ディレクトリを作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 分割した設定ファイルを保存
    for i, split in enumerate(splits):
        output_file = Path(output_dir) / f"config_{i}.json"
        with open(output_file, "w") as f:
            json.dump(split, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_config.py <config_file> <output_dir> <n_splits>")
        sys.exit(1)

    config_file = sys.argv[1]
    output_dir = sys.argv[2]
    n_splits = int(sys.argv[3])

    split_config(config_file, output_dir, n_splits)
