import math
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Union
import json
from tqdm import tqdm
import numpy as np
from functools import partial

from nltk.lm import Lidstone, MLE
from nltk.util import ngrams, pad_sequence
from local_entropy.ngram_util import padded_everygram_pipeline, pad_both_ends, pad_eos


def load_data(train_path: Path, valid_path: Path, test_path: Path) -> List[str]:
    """
    Load sentences from specified files for train, validation, and test.
    Each line is treated as a sentence.
    Returns a list of sentences.
    """
    all_sentences = []

    paths = [
        (name, path)
        for name, path in zip(
            ["train", "valid", "test"], [train_path, valid_path, test_path]
        )
        if path.exists()
    ]
    print(f"Loading {len(paths)} datasets...", flush=True)

    for name, path in tqdm(paths, desc="Loading files"):
        with path.open("r", encoding="utf-8") as f:
            sentences = f.read().splitlines()
        all_sentences.extend(sentences)
    return all_sentences


def tokenize_corpus(sentences: List[str]) -> List[List[str]]:
    """
    Tokenize all sentences in the corpus.

    Args:
        sentences: List of sentences to tokenize

    Returns:
        List of tokenized sentences
    """
    return [sentence.split() for sentence in tqdm(sentences, desc="Tokenizing")]


def train_ngram_model(
    n: int,
    tokenized_text: List[List[str]],
    method: str = "MLE",
    gamma: float = None,
    add_bos: bool = False,
) -> Union[MLE, Lidstone]:
    """
    Train an n-gram model on the provided corpus.

    Args:
        n: The order of n-gram model
        tokenized_text: List of tokenized sentences
        method: Estimation method ('MLE', 'Laplace', or 'Lidstone')
        gamma: Lidstone smoothing parameter (only used when method='Lidstone')
        add_bos: Whether to add <bos> to the beginning of each sentence

    Returns:
        Language model (MLE or Lidstone)
    """
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text, add_bos)

    if method == "MLE":
        model = MLE(n)
    elif method == "Laplace":
        model = Lidstone(1.0, n)  # Laplace smoothing is Lidstone with gamma=1
    elif method == "Lidstone":
        if gamma is None:
            raise ValueError("gamma parameter must be specified for Lidstone smoothing")
        model = Lidstone(gamma, n)
    else:
        raise ValueError(f"Unknown estimation method: {method}")

    model.fit(train_data, padded_vocab)
    del train_data, padded_vocab
    return model


def calculate_entropy_nltk(
    model: Union[MLE, Lidstone],
    tokenized_sentences: List[List[str]],
    n: int,
    add_bos: bool = False,
) -> float:
    """
    Calculate the entropy (average negative log probability in bits) of the corpus
    using the trained n-gram model.

    Args:
        model: Trained n-gram language model
        tokenized_sentences: List of tokenized sentences
        n: The order of n-gram model

    Returns:
        float: Entropy value in bits
    """
    total_log_prob = 0.0
    total_ngrams = 0

    for sentence in tqdm(tokenized_sentences, desc="Calculating entropy"):
        if add_bos:
            sentence_with_special_tokens = pad_both_ends(sentence, n)
        else:
            sentence_with_special_tokens = pad_eos(sentence)

        for ngram in ngrams(sentence_with_special_tokens, n):
            prob = model.score(ngram[-1], ngram[:-1])
            if prob == 0:
                print(ngram)
            total_log_prob += np.log2(prob)
            total_ngrams += 1

    entropy = -1 * (total_log_prob / total_ngrams)
    return entropy


def main():
    parser = argparse.ArgumentParser(
        description="Calculate n-gram entropy using various estimation methods "
        + "on the combined (train+valid+test) corpus."
    )
    parser.add_argument(
        "--train_path", type=Path, required=True, help="Path to training data file"
    )
    parser.add_argument(
        "--valid_path", type=Path, required=True, help="Path to validation data file"
    )
    parser.add_argument(
        "--test_path", type=Path, required=True, help="Path to test data file"
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["MLE", "Laplace", "Lidstone"],
        default="MLE",
        help="Probability estimation method (default: MLE)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Smoothing parameter for Lidstone estimation (required if method=Lidstone)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="n-gram sizes (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--add_bos",
        action="store_true",
        help="Add <bos> to the beginning of each sentence",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.method == "Lidstone" and args.gamma is None:
        parser.error("--gamma is required when using Lidstone estimation")
    if args.method != "Lidstone" and args.gamma is not None:
        parser.error("--gamma can only be used with Lidstone estimation")

    # Force flush the output buffer
    print("Loading datasets...", flush=True)
    corpus = load_data(args.train_path, args.valid_path, args.test_path)

    print("Tokenizing corpus...", flush=True)
    tokenized_corpus = tokenize_corpus(corpus)
    del corpus

    # Combine all data for training and evaluation
    results = {"local_entropy": {}}

    for n in args.n:
        print(f"Processing {n}-gram model on the combined corpus...", flush=True)
        print(f"Training model ({args.method})...", flush=True)
        model = train_ngram_model(
            n, tokenized_corpus, args.method, args.gamma, args.add_bos
        )

        print("Calculating entropy...", flush=True)
        entropy = calculate_entropy_nltk(model, tokenized_corpus, n, args.add_bos)
        results["local_entropy"][f"{n}"] = entropy
        print(f"{n}_local_entropy: {entropy}", flush=True)

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
