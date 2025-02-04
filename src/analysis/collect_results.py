import json
import pandas as pd
from pathlib import Path
import re
import argparse


def extract_grammar_and_trial(path):
    # パスから文法名とtrial番号を抽出
    match = re.match(r".*?/([^/]+)_trial(\d+)/evaluation/test.json", str(path))
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def collect_results(data_dir, result_dir, exp_name, architectures, output_path):
    results = []

    for arch in architectures:
        # 評価結果のディレクトリパス
        result_base = Path(result_dir) / exp_name / arch

        # test.jsonファイルを検索
        for test_path in result_base.glob("**/test.json"):
            grammar_name, trial = extract_grammar_and_trial(test_path)
            if grammar_name is None:
                continue

            # メタデータのパス
            metadata_path = Path(data_dir) / exp_name / grammar_name / "metadata.json"

            try:
                # テスト結果の読み込み
                with open(test_path) as f:
                    test_results = json.load(f)

                # メタデータの読み込み
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # 結果の辞書を作成
                result = {
                    "grammar_name": grammar_name,
                    "trial": trial,
                    "architecture": arch,
                    "n_states": metadata["n_states"],
                    "N_sym": metadata["N_sym"],
                    "topology_seed": metadata["topology_seed"],
                    "weight_seed": metadata["weight_seed"],
                    "mean_length": metadata["mean_length"],
                    "entropy": metadata["entropy"],
                    "next_symbol_entropy": metadata["next_symbol_entropy"],
                    "2_local_entropy": metadata["local_entropy"]["2"],
                    "3_local_entropy": metadata["local_entropy"]["3"],
                    "4_local_entropy": metadata["local_entropy"]["4"],
                    "5_local_entropy": metadata["local_entropy"]["5"],
                    "cross_entropy_per_token": test_results["cross_entropy_per_token"],
                    "perplexity": test_results["perplexity"],
                    "cross_entropy_per_token_base_e": test_results[
                        "cross_entropy_per_token_base_e"
                    ],
                    "cross_entropy_per_token_base_2": test_results[
                        "cross_entropy_per_token_base_2"
                    ],
                }

                results.append(result)

            except FileNotFoundError as e:
                print(f"Warning: Could not find file: {e.filename}")
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in file: {e.doc}")
            except KeyError as e:
                print(f"Warning: Missing key in data: {e}")

    # DataFrameを作成して保存
    if results:
        df = pd.DataFrame(results)
        # 列の順序を指定
        columns = [
            "grammar_name",
            "trial",
            "architecture",
            "n_states",
            "N_sym",
            "topology_seed",
            "weight_seed",
            "mean_length",
            "entropy",
            "next_symbol_entropy",
            "2_local_entropy",
            "3_local_entropy",
            "4_local_entropy",
            "5_local_entropy",
            "cross_entropy_per_token",
            "perplexity",
            "cross_entropy_per_token_base_e",
            "cross_entropy_per_token_base_2",
        ]
        df = df[columns]
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results found")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the PFSA data"
    )
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Directory containing the results"
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--architectures", nargs="+", required=True, help="List of architectures"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the collected results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_results(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        exp_name=args.exp_name,
        architectures=args.architectures,
        output_path=args.output_path,
    )
