import sys

sys.path.append("..")
from src.length_sampling.sampler import construct_pcfg_sampler
from src.length_sampling.grammars.pcfg import Grammar
from src.length_sampling.grammars.cfg import Nonterminal
from src.length_sampling.util import group_by, get_random_generator_and_seed
import argparse
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import json
import gzip
from src.length_sampling.lower_bound_perplexity import (
    parts_to_perplexity,
    Parts,
)
import math
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_file", type=str)
    parser.add_argument("--start_symbol", type=str, default="S")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--exp_dir", type=lambda p: Path(p).resolve())
    parser.add_argument("--input_suffix", type=str, default="json.gz")
    args = parser.parse_args()
    grammar = Grammar.from_file(
        args.grammar_file, Nonterminal(args.start_symbol), args.normalize
    )
    sampler = construct_pcfg_sampler(grammar)

    print(f"Reading sentences from {args.exp_dir}...")
    dfs = []
    for file in args.exp_dir.glob(f"*{args.input_suffix}"):
        with gzip.open(file, "rt") as f:
            dfs.append(pd.read_json(f, lines=True, orient="records"))

    print(f"{len(dfs)} files found.")
    df = pd.concat(dfs, ignore_index=True)

    # calculate lower bound perplexity
    valid_lengths = sampler.valid_lengths(args.min_length, args.max_length)
    total_neg_log_prob = -1.0 * df["true_log_prob"].sum()

    df["sent_len"] = df["sentence"].map(lambda x: len(x.split()))
    total_len = (df["sent_len"] * df["count"]).sum()
    num_samples = df["count"].sum()
    parts = Parts(total_neg_log_prob, total_len, num_samples)
    perplexity = parts_to_perplexity(parts, len(valid_lengths))
    entropy = math.log(perplexity)

    # save
    with open(args.exp_dir / "lower_bound_perplexity.value", "w") as f:
        f.write(str(perplexity))

    with open(args.exp_dir / "lower_bound_entropy.value", "w") as f:
        f.write(str(entropy))


if __name__ == "__main__":
    main()
