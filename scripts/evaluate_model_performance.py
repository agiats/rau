import math
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def evaluate_models(exp_name, model_names=None, num_seeds=5):
    """
    Evaluate model performance across different grammars and seeds.

    Args:
        exp_name (str): Name of the experiment
        model_names (list): List of model names to evaluate. Defaults to ["lstm", "transformer_4layer"]
        num_seeds (int): Number of random seeds used in training. Defaults to 5

    Returns:
        pd.DataFrame: DataFrame containing evaluation results
    """
    if model_names is None:
        model_names = ["lstm", "transformer_4layer"]

    data_dir = Path(f"../data/fairseq_train/{exp_name}")
    results_dir = Path("../results").resolve()

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

    # Save results to CSV
    output_file = results_dir / f"length_sampling_{exp_name}_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    return result_df


if __name__ == "__main__":
    # Example usage
    exp_name = "local_entropy_prelim"
    result_df = evaluate_models(exp_name)
