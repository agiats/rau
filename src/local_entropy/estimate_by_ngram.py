import math
import argparse
import pandas as pd
from pathlib import Path
from typing import List
import json
from tqdm import tqdm

# --- Import NLTK modules ---
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends


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


def train_ngram_model(n: int, sentences: list, add_eos: bool) -> MLE:
    """
    Train an n-gram model using MLE (Maximum Likelihood Estimation) on the provided corpus.

    If add_eos is True, use NLTK's preprocessing to automatically add BOS (<s>) and EOS (</s>) tokens to each sentence.
    If False, no padding is applied and n-grams are generated directly from the tokenized sentences.

    The MLE model computes probabilities as:
        P(w | h) = count(h, w) / count(h)
    This is a pure count-based estimation without any smoothing or backoff.
    """
    # Tokenize each sentence by splitting on whitespace
    tokenized_text = [sentence.split() for sentence in sentences]

    if add_eos:
        # Use padded_everygram_pipeline to add BOS and EOS tokens automatically
        train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    else:
        # Without padding: directly generate n-grams from each sentence (only if sentence length >= n)
        train_data = (
            [tuple(sentence[i : i + n]) for i in range(len(sentence) - n + 1)]
            for sentence in tokenized_text
        )
        # The vocabulary is the set of all tokens in the corpus
        vocab_set = set(token for sentence in tokenized_text for token in sentence)
        padded_vocab = list(vocab_set)

    # MLE model uses pure counts for probability estimation (no smoothing or backoff)
    model = MLE(n)
    model.fit(train_data, padded_vocab)
    return model


def calculate_entropy_nltk(model: MLE, sentences: list, n: int, add_eos: bool) -> float:
    """
    Calculate the entropy (average negative log probability in bits) of the corpus
    using the trained n-gram model.

    If add_eos is True, each sentence is padded with BOS and EOS tokens using pad_both_ends.
    If False, n-grams are generated from the tokenized sentence without padding.

    Note: Since the MLE model is based solely on counts, any unseen n-gram will have a probability of 0,
    causing the log probability to be -∞ and the overall entropy to be infinity.
    """
    total_log_prob = 0.0
    total_ngrams = 0

    for sentence in tqdm(sentences, desc="Calculating entropy"):
        tokens = sentence.split()
        if add_eos:
            # Add BOS and EOS tokens. For example, for n=3:
            # ['<s>', '<s>', token1, token2, ..., tokenN, '</s>']
            padded_sentence = list(pad_both_ends(tokens, n))
            for i in range(n - 1, len(padded_sentence) - n + 2):
                context = tuple(padded_sentence[i - n + 1 : i])
                word = padded_sentence[i]
                prob = model.score(word, context)
                if prob == 0:
                    print(f"prob is 0: {word} {context}")
                    return float("inf")
                total_log_prob += math.log(prob, 2)  # log base 2
                total_ngrams += 1
        else:
            # Without padding, generate n-grams directly from the tokenized sentence
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                # For n=1, the context is an empty tuple
                context = tuple(tokens[i : i + n - 1]) if n > 1 else ()
                word = tokens[i + n - 1]
                prob = model.score(word, context)
                if prob == 0:
                    return float("inf")
                total_log_prob += math.log(prob, 2)
                total_ngrams += 1

    if total_ngrams == 0:
        return float("inf")
    # Entropy in bits is the negative average log probability
    return -total_log_prob / total_ngrams


def main():
    parser = argparse.ArgumentParser(
        description="Calculate n-gram entropy using NLTK MLE (pure count; no smoothing/backoff) "
        + "on the combined (train+valid+test) corpus."
    )
    parser.add_argument("train_path", type=Path, help="Path to training data file")
    parser.add_argument("valid_path", type=Path, help="Path to validation data file")
    parser.add_argument("test_path", type=Path, help="Path to test data file")
    parser.add_argument("output_path", type=Path, help="Path to output CSV file")
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="n-gram sizes (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--no-eos",
        action="store_false",
        dest="add_eos",
        help="Do not add BOS/EOS tokens (no padding)",
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
        model = train_ngram_model(n, corpus, args.add_eos)
        print("Calculating entropy...")
        entropy = calculate_entropy_nltk(model, corpus, n, args.add_eos)
        results["local_entropy"][f"{n}"] = entropy
        print(f"{n}_local_entropy: {entropy}")

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
