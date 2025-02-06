import json
import pandas as pd
from pathlib import Path
import re
import argparse


def extract_grammar_and_trial(path, split_name):
    pattern = rf".*?/([^/]+)_trial(\d+)/evaluation/{split_name}\.json"
    match = re.match(pattern, str(path))
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def collect_results(
    data_dir, result_base_dir, exp_name, architectures, split_name, output_path
):
    results = []

    for arch in architectures:
        result_base = Path(result_base_dir) / exp_name / arch
        print(f"Collecting results for {arch} in {result_base}")
        for test_path in result_base.glob(f"*/evaluation/{split_name}.json"):
            grammar_name, trial = extract_grammar_and_trial(test_path, split_name)
            if grammar_name is None:
                continue

            metadata_path = Path(data_dir) / exp_name / grammar_name / "metadata.json"

            try:

                with open(test_path) as f:
                    test_results = json.load(f)

                with open(metadata_path) as f:
                    metadata = json.load(f)

                result = {
                    "grammar_name": grammar_name,
                    "trial": trial,
                    "architecture": arch,
                    "n_states": (
                        metadata["n_states"] if "n_states" in metadata else None
                    ),
                    "N_sym": metadata["N_sym"] if "N_sym" in metadata else None,
                    "topology_seed": (
                        metadata["topology_seed"]
                        if "topology_seed" in metadata
                        else None
                    ),
                    "weight_seed": (
                        metadata["weight_seed"] if "weight_seed" in metadata else None
                    ),
                    "mean_length": (
                        metadata["mean_length"] if "mean_length" in metadata else None
                    ),
                    "entropy": metadata["entropy"] if "entropy" in metadata else None,
                    "next_symbol_entropy": (
                        metadata["next_symbol_entropy"]
                        if "next_symbol_entropy" in metadata
                        else None
                    ),
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

    if results:
        df = pd.DataFrame(results)
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
        "--result_base_dir",
        type=str,
        required=True,
        help="Directory containing the results",
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--architectures", nargs="+", required=True, help="List of architectures"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        required=True,
        help="Split name to collect results for",
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
        result_base_dir=args.result_base_dir,
        exp_name=args.exp_name,
        architectures=args.architectures,
        split_name=args.split_name,
        output_path=args.output_path,
    )
