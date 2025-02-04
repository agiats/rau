import math
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse


def evaluate_models(
    exp_name,
    data_dir,
    results_dir,
    output_dir=None,
    model_names=None,
    num_seeds=5,
):
    """
    Evaluate model performance across different grammars and seeds.

    Args:
        exp_name (str): Name of the experiment
        data_dir (Path): Directory containing fairseq training data
        results_dir (Path): Directory containing model results
        output_dir (Path): Directory to save evaluation results. Defaults to results_dir
        model_names (list): List of model names to evaluate. Defaults to ["lstm", "transformer_4layer"]
        num_seeds (int): Number of random seeds used in training. Defaults to 5

    Returns:
        pd.DataFrame: DataFrame containing evaluation results
    """
    if model_names is None:
        model_names = ["lstm", "transformer_4layer"]

    if output_dir is None:
        output_dir = results_dir

    data_dir = Path(data_dir) / exp_name
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)

    # Get list of grammar names from directories
    grammar_names = [d.name for d in data_dir.iterdir() if d.is_dir()]

    result_list = []
    for grammar_name in tqdm(grammar_names):
        # Calculate total symbols from test data
        split_data_file = data_dir / grammar_name / "test.txt"
        with open(split_data_file) as f:
            total_syms = sum(
                len(line.strip().split()) + 1 for line in f
            )  # add 1 for EOS token

        for model_name in model_names:
            model_result_dir = results_dir / f"{model_name}_results" / exp_name
            grammar_result_dir = model_result_dir / f"{grammar_name}"

            for seed_i in range(num_seeds):
                # Read model scores
                split_result_file = (
                    grammar_result_dir / f"seed{seed_i}" / "test.scores.txt"
                )
                with open(split_result_file) as f:
                    scores = [float(line.strip()) for line in f]

                # Calculate metrics
                total_sents = len(scores)
                neg_log_probs = [-1.0 * score for score in scores]

                # Symbol-level metrics
                sym_cross_entropy = (sum(neg_log_probs) / total_syms) / math.log(2)
                sym_perplexity = math.exp(sym_cross_entropy)

                # Sentence-level metrics
                sent_cross_entropy = (sum(neg_log_probs) / total_sents) / math.log(2)
                sent_perplexity = math.exp(sent_cross_entropy)

                # Store results
                result_list.append(
                    {
                        "model_name": model_name,
                        "seed": seed_i,
                        "grammar_name": grammar_name,
                        "sym_cross_entropy": sym_cross_entropy,
                        "sym_perplexity": sym_perplexity,
                        "sent_cross_entropy": sent_cross_entropy,
                        "sent_perplexity": sent_perplexity,
                    }
                )

        print(f"Grammar: {grammar_name}")
        print(f"Total symbols: {total_syms}")
        print(f"Total sentences: {total_sents}")
        print(f"Average symbols per sentence: {total_syms / total_sents:.2f}\n")

    # Create and save results DataFrame
    result_df = (
        pd.DataFrame(result_list)
        .sort_values(["model_name", "grammar_name"])
        .reset_index(drop=True)
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    output_file = output_dir / f"length_sampling_{exp_name}_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    return result_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model performance across different grammars and seeds."
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fairseq_train",
        help="Directory containing fairseq training data (default: ../data/fairseq_train)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing model results (default: ../results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save evaluation results (default: same as results-dir)",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["lstm", "transformer_4layer"],
        help="List of model names to evaluate (default: lstm transformer_4layer)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of random seeds used in training (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_df = evaluate_models(
        exp_name=args.exp_name,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        model_names=args.model_names,
        num_seeds=args.num_seeds,
    )
