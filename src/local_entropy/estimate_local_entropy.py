import math
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Union
import json
from tqdm import tqdm
import numpy as np

from nltk.lm import Lidstone, MLE
from nltk.util import ngrams
from local_entropy.ngram_util import eos_ngram_pipeline


def load_data(train_path: Path, valid_path: Path, test_path: Path) -> List[str]:
    """
    Load sentences from specified files for train, validation, and test.
    Each line is treated as a sentence.
    Returns a list of sentences.
    """
    all_sentences = []

    for name, path in zip(
        ["train", "valid", "test"], [train_path, valid_path, test_path]
    ):
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                sentences = f.read().splitlines()
            all_sentences.extend(sentences)
    return all_sentences


def train_ngram_model(n: int, sentences: list, gamma: int = 0) -> Union[MLE, Lidstone]:
    """
    Train an n-gram model on the provided corpus.
    Uses MLE when gamma=0, otherwise uses Lidstone smoothing.

    Args:
        n: The order of n-gram model
        sentences: List of sentences to train on
        gamma: Lidstone smoothing parameter (default=0.0 for MLE)

    Returns:
        MLE model if gamma=0, otherwise Lidstone model
    """
    tokenized_text = [sentence.split() for sentence in sentences]

    # evergramsの代わりに固定長のn-gramを使用
    train_data, padded_vocab = eos_ngram_pipeline(n, tokenized_text)

    if gamma == 0:
        model = MLE(n)
    else:
        model = Lidstone(gamma, n)
    model.fit(train_data, padded_vocab)
    return model


def calculate_entropy_nltk(
    model: Union[MLE, Lidstone], sentences: list, n: int, add_eos: bool = True
) -> float:
    """
    Calculate the entropy (average negative log probability in bits) of the corpus
    using the trained n-gram model.
    """

    tokenized_sentences = [sentence.split() + ["</s>"] for sentence in sentences]
    total_log_prob = 0.0
    total_ngrams = 0

    for sentence in tokenized_sentences:
        for ngram in ngrams(sentence, n):
            total_log_prob += np.log2(model.score(ngram[-1], ngram[:-1]))
            total_ngrams += 1

    entropy = -1 * (total_log_prob / total_ngrams)
    return entropy


def main():
    parser = argparse.ArgumentParser(
        description="Calculate n-gram entropy using MLE or Lidstone smoothing "
        + "on the combined (train+valid+test) corpus."
    )
    parser.add_argument("--train_path", type=Path, help="Path to training data file")
    parser.add_argument("--valid_path", type=Path, help="Path to validation data file")
    parser.add_argument("--test_path", type=Path, help="Path to test data file")
    parser.add_argument("--output_path", type=Path, help="Path to output CSV file")
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="n-gram sizes (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=0,
        help="Lidstone smoothing parameter (default: 0 for MLE)",
    )
    args = parser.parse_args()

    print("Loading datasets...")
    corpus = load_data(args.train_path, args.valid_path, args.test_path)

    # Combine all data for training and evaluation
    results = {"local_entropy": {}}

    for n in args.n:
        print(f"Processing {n}-gram model on the combined corpus...")
        # Train the model using MLE (pure count-based probability estimation)
        print("Training model...")
        model = train_ngram_model(n, corpus, args.gamma)
        print("Calculating entropy...")
        entropy = calculate_entropy_nltk(model, corpus, n)
        results["local_entropy"][f"{n}"] = entropy
        print(f"{n}_local_entropy: {entropy}")

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
