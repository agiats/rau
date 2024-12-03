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
from functools import partial
import multiprocessing as mp


def process_sentence_with_sampler(sampler, sentence: str):
    """Helper function for multiprocessing that calculates log probability for a single sentence."""
    log_prob = sampler.log_probability_given_length(sentence.split())
    return sentence, float(log_prob)


def calculate_true_log_probabilities(sampler, sentences, num_workers):
    """Calculate true log probabilities for given sentences.

    Args:
        sampler: PCFG sampler object with log_probability_given_length method
        sentences: list of sentences to calculate probabilities for
        num_workers: number of worker processes for parallel computation

    Returns:
        dict: mapping from sentences to their true log probabilities
    """
    unique_sentences = set(sentences)  # Remove duplicates

    if num_workers == 1:
        # Direct processing without parallel execution
        return {
            sentence: prob
            for sentence, prob in map(
                partial(process_sentence_with_sampler, sampler),
                tqdm(unique_sentences, desc="Calculating true probabilities"),
            )
        }

    # Initialize multiprocessing Pool with shared sampler
    with mp.Pool(
        processes=num_workers, initializer=init_worker, initargs=(sampler,)
    ) as pool:
        results = {}
        # Use imap_unordered for better progress tracking
        for sentence, prob in tqdm(
            pool.imap_unordered(process_sentence_worker, unique_sentences),
            total=len(unique_sentences),
            desc="Calculating true probabilities",
        ):
            results[sentence] = prob

    return results


def init_worker(sampler_instance):
    """Initialize each worker process with a copy of the sampler."""
    global shared_sampler
    shared_sampler = sampler_instance


def process_sentence_worker(sentence):
    """Worker function that uses the shared sampler."""
    global shared_sampler
    return process_sentence_with_sampler(shared_sampler, sentence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_file", type=str)
    parser.add_argument("--start_symbol", type=str, default="S")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--sentence_counts_path", type=lambda p: Path(p).resolve())
    parser.add_argument("--output_path", type=lambda p: Path(p).resolve())
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel sampling",
    )
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    args = parser.parse_args()
    grammar = Grammar.from_file(
        args.grammar_file, Nonterminal(args.start_symbol), args.normalize
    )
    sampler = construct_pcfg_sampler(grammar)

    # load sentence counts
    print("Loading sentence counts")
    sentence_counts = pl.read_csv(
        args.sentence_counts_path, new_columns=["sentence", "count"]
    )
    sentence_counts = sentence_counts.filter(pl.col("count").is_not_null())

    if args.start_index is not None and args.end_index is not None:
        sentence_counts = sentence_counts[args.start_index : args.end_index]

    print("Calculating true probabilities...")
    sentences = sentence_counts["sentence"].to_list()
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

    # output
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_path, "wt") as f:
        sentence_counts.write_csv(f, include_header=True)


if __name__ == "__main__":
    main()
