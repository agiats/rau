import sys

sys.path.append("..")
from src.length_sampling.sampler import construct_pcfg_sampler
from src.length_sampling.grammars.pcfg import Grammar
from src.length_sampling.grammars.cfg import Nonterminal
import argparse
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import polars as pl
from src.length_sampling.lower_bound_perplexity import (
    parts_to_perplexity,
    Parts,
)
import gzip
import math
import json


def calculate_true_log_probabilities(sampler, sentences, num_workers):
    """Calculate true log probabilities for given sentences.

    Args:
        sampler: PCFG sampler object with log_probability_given_length method
        sentences: list of sentences to calculate probabilities for
        num_workers: number of worker processes for parallel computation

    Returns:
        dict: mapping from sentences to their true log probabilities
    """

    def process_sentence(sentence: str):
        log_prob = sampler.log_probability_given_length(sentence.split())
        return sentence, float(log_prob)

    unique_sentences = set(sentences)  # Remove duplicates

    if num_workers == 1:
        # Direct processing without parallel execution
        return {
            sentence: prob
            for sentence, prob in map(
                process_sentence,
                tqdm(unique_sentences, desc="Calculating true probabilities"),
            )
        }

    # Parallel processing for num_workers > 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_sentence = {
            executor.submit(process_sentence, sentence): sentence
            for sentence in unique_sentences
        }

        results = {}
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sentence),
            total=len(future_to_sentence),
            desc="Calculating true probabilities",
        ):
            sentence, prob = future.result()
            results[sentence] = prob

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_file", type=str)
    parser.add_argument("--start_symbol", type=str, default="S")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--sample_path", type=lambda p: Path(p).resolve())
    parser.add_argument("--output_path", type=lambda p: Path(p).resolve())
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel sampling",
    )
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    args = parser.parse_args()
    grammar = Grammar.from_file(
        args.grammar_file, Nonterminal(args.start_symbol), args.normalize
    )
    sampler = construct_pcfg_sampler(grammar)

    print("Reading sentences...")
    if args.sample_size > 0:
        # Only read the first sample_size lines to save memory
        with gzip.open(args.sample_path, "rt", encoding="utf-8") as f:
            sentences = [next(f).strip() for _ in range(args.sample_size)]
        print("Sampling size:", len(sentences))
    else:
        with gzip.open(args.sample_path, "rt", encoding="utf-8") as f:
            sentences = f.read().strip().split("\n")

    if args.start_index is not None and args.end_index is not None:
        sentences = sentences[args.start_index : args.end_index]

    counter = Counter(sentences)
    sentence_counts = pl.DataFrame(
        {
            "sentence": list(counter.keys()),
            "count": list(counter.values()),
        }
    )

    print("Calculating true probabilities...")
    sentence_to_true_log_prob = calculate_true_log_probabilities(
        sampler, sentences, args.num_workers
    )
    sentence_to_true_log_prob = pl.DataFrame(
        {
            "sentence": list(sentence_to_true_log_prob.keys()),
            "true_log_prob": list(sentence_to_true_log_prob.values()),
            "true_prob": [math.exp(x) for x in sentence_to_true_log_prob.values()],
        }
    )
    # merge
    sentence_counts = sentence_counts.join(
        sentence_to_true_log_prob, on="sentence", how="inner"
    )
    assert len(sentence_counts) == len(counter)

    # output
    with gzip.open(args.output_path, "wt", encoding="utf-8") as f:
        for row in sentence_counts.iter_rows(named=True):
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
