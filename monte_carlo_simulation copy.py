import argparse
import multiprocessing as mp
from pathlib import Path
import json
from collections import Counter
from functools import partial
import tqdm
import argparse
import random
from pcfg import PCFG
from hydra.utils import instantiate



def sample_batch(args):
    grammar_class, grammar_file, max_expansions, batch_size, sub_batch_size = args
    grammar = instantiate({"_target_": f"pcfg.{grammar_class}", "grammar_file": grammar_file})
    sentence_counts = Counter()
    samples_generated = 0

    while samples_generated < batch_size:
        current_size = min(sub_batch_size, batch_size - samples_generated)
        sub_counter = Counter()
        for _ in range(current_size):
            sent = grammar.sample_sentence(max_expansions=max_expansions, bracketing=False)
            if sent is not None:
                sub_counter[sent] += 1
                samples_generated += 1
        sentence_counts.update(sub_counter)
        del sub_counter

    return sentence_counts

def monte_carlo_parallel(grammar_class, grammar_file, max_expansions, n_samples, n_processes=None, sub_batch_size=10000):
    if n_processes is None:
        n_processes = mp.cpu_count()

    base_batch_size = n_samples // n_processes
    remainder = n_samples % n_processes

    batch_args = []
    for i in range(n_processes):
        batch_size = base_batch_size + (1 if i < remainder else 0)
        batch_args.append((grammar_class, grammar_file, max_expansions, batch_size, sub_batch_size))

    with mp.Pool(n_processes) as pool:
        total_counts = Counter()
        total_samples = 0  # 実際のサンプル数をカウント
        with tqdm.tqdm(total=n_samples, desc="Sampling sentences") as pbar:
            for batch_counts in pool.imap_unordered(sample_batch, batch_args):
                total_counts.update(batch_counts)
                batch_samples = sum(batch_counts.values())
                total_samples += batch_samples
                pbar.update(batch_samples)

        if total_samples != n_samples:
            print(f"Warning: Requested {n_samples} samples but got {total_samples}")

    sentence_probs = {k: v/total_samples for k, v in total_counts.items()}

    return total_counts, sentence_probs, total_samples  # 実際のサンプル数も返す


def main():
    parser = argparse.ArgumentParser(description="Parallel Monte Carlo sampling for PCFG")
    parser.add_argument("--grammar_class", type=str, help="Grammar class")
    parser.add_argument("--grammar_file", type=str, help="Path to grammar file")
    parser.add_argument("--n_samples", type=int, default=1_000_000, help="Number of samples")
    parser.add_argument("--sub_batch_size", type=int, default=10_000, help="Sub-batch size")
    parser.add_argument("--max_expansions", type=int, default=100, help="Max number of expansions")
    parser.add_argument("--n_processes", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # モンテカルロサンプリングの実行
    counts, probs, actual_samples = monte_carlo_parallel(
        args.grammar_class,
        args.grammar_file,
        args.max_expansions,
        args.n_samples,
        args.n_processes,
        sub_batch_size=args.sub_batch_size
    )


    # 結果の保存
    with open(output_dir / "sentence_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

    with open(output_dir / "sentence_probs.json", "w") as f:
        json.dump(probs, f, indent=2)

    print(f"\nRequested samples: {args.n_samples}")
    print(f"Actual samples: {actual_samples}")
    print(f"Total unique sentences: {len(counts)}")
    print(f"Most common sentence: {max(counts.items(), key=lambda x: x[1])}")
    print(f"Least common sentence: {min(counts.items(), key=lambda x: x[1])}")

if __name__ == "__main__":
    main()
# python monte_carlo_simulation.py --grammar_class PCFG --grammar_file data_gen/base-grammar_eos.gr --n_samples 10000 --max_expansions 400 --n_processes 15 --output_dir results
# python monte_carlo_simulation.py --grammar_class PCFGDeterministicShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 10000 --max_expansions 400 --n_processes 15 --output_dir results
# python monte_carlo_simulation.py --grammar_class PCFGNonDeterministicShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 10000 --max_expansions 400 --n_processes 15 --output_dir results
# python monte_carlo_simulation.py --grammar_class PCFGLocalShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 10000 --max_expansions 400 --n_processes 15 --output_dir results
# python monte_carlo_simulation.py --grammar_class PCFGEvenOddShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 10000 --max_expansions 400 --n_processes 15 --output_dir results
