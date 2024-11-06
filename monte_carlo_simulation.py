import argparse
import multiprocessing as mp
from pathlib import Path
import json
from collections import Counter
from functools import partial
import tqdm
import time
import argparse
import random
import concurrent.futures
from collections import Counter
import multiprocessing as mp
from pcfg import PCFG
from hydra.utils import instantiate
import gzip
import numpy as np



def sample_batch(args):
    grammar_class, grammar_file, max_expansions, batch_size, process_id = args
    grammar = instantiate({"_target_": f"pcfg.{grammar_class}", "grammar_file": grammar_file})
    sentence_counts = Counter()
    successful_samples = 0

    # バッチ内でサンプリング
    for _ in range(batch_size):
        sent = grammar.sample_sentence(max_expansions=max_expansions, bracketing=False)
        if sent is not None:
            sentence_counts[sent] += 1
            successful_samples += 1

    return process_id, successful_samples, sentence_counts

def monte_carlo_parallel(grammar_class, grammar_file, max_expansions, n_samples, batch_size=10000, n_processes=None):
    if n_processes is None:
        n_processes = mp.cpu_count()

    total_counts = Counter()
    total_samples = 0
    start_time = time.time()

    # メインのプログレスバーを初期化
    with tqdm.tqdm(total=n_samples, desc="Total Progress") as pbar:
        pbar.set_postfix({
            'unique': 0,
            'samples/s': 0,
            'processes': n_processes,
            'batch': 0
        })

        with mp.Pool(n_processes) as pool:
            remaining_samples = n_samples
            batch_count = 0

            while remaining_samples > 0:
                batch_count += 1
                current_batch = min(batch_size * n_processes, remaining_samples)
                samples_per_process = current_batch // n_processes
                remainder = current_batch % n_processes

                # バッチ引数の準備
                batch_args = []
                for i in range(n_processes):
                    process_batch_size = samples_per_process + (1 if i < remainder else 0)
                    if process_batch_size > 0:
                        batch_args.append((grammar_class, grammar_file, max_expansions, process_batch_size, i))

                # バッチ処理の実行
                for process_id, samples, counts in pool.imap_unordered(sample_batch, batch_args):
                    total_counts.update(counts)
                    total_samples += samples
                    remaining_samples -= samples

                    # プログレスバーの更新
                    elapsed = time.time() - start_time
                    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'unique': len(total_counts),
                        'samples/s': f'{samples_per_sec:.0f}',
                        'processes': n_processes,
                        'batch': batch_count
                    })
                    pbar.update(samples)

    return total_counts, total_samples

def main():
    parser = argparse.ArgumentParser(description="Parallel Monte Carlo sampling for PCFG")
    parser.add_argument("--grammar_class", type=str, help="Grammar class")
    parser.add_argument("--grammar_file", type=str, help="Path to grammar file")
    parser.add_argument("--n_samples", type=int, default=1_000_000, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Sub-batch size")
    parser.add_argument("--max_expansions", type=int, default=100, help="Max number of expansions")
    parser.add_argument("--n_processes", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--output_name", type=str, default="samples", help="Output file name")

    args = parser.parse_args()

    counts, actual_samples = monte_carlo_parallel(
        args.grammar_class,
        args.grammar_file,
        args.max_expansions,
        args.n_samples,
        batch_size=args.batch_size,
        n_processes=args.n_processes
    )

    output_dir = Path(args.output_dir) / f"{args.output_name}" / f"{args.grammar_class}"
    output_dir.mkdir(exist_ok=True, parents=True)

    counts_file = output_dir / f"sentence_counts.gz"
    with gzip.open(counts_file, 'wt', encoding='utf-8') as f:
        for sentence, count in counts.items():
            f.write(f"{sentence}\t{count}\n")

    # calculate entropy (using numpy)
    Epsilon = 1e-10
    np_counts = np.array(list(counts.values()))
    probs = np_counts / actual_samples
    entropy = -np.sum(probs * np.log2(probs + Epsilon))


    # results
    results_dict = {
        "Entropy": entropy,
        "Requested samples": args.n_samples,
        "Actual samples": actual_samples,
        "Total unique sentences": len(counts),
        "Most common sentence": max(counts.items(), key=lambda x: x[1]),
        "Least common sentence": min(counts.items(), key=lambda x: x[1])
    }

    for k, v in results_dict.items():
        print(f"{k}: {v}")

    results_file = output_dir / f"results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)



if __name__ == "__main__":
    main()
# python monte_carlo_simulation.py data_gen/base-grammar_eos.gr --n_samples 1_000_000_000 --max_expansions 400 --n_processes 15 --output_dir results --output_name 1G_samples --batch_size 100000
# python monte_carlo_simulation.py --grammar_class PCFG --grammar_file data_gen/base-grammar_eos.gr --n_samples 100000 --max_expansions 400 --n_processes 15 --output_dir results --output_name 1G_samples
# python monte_carlo_simulation.py --grammar_class PCFGDeterministicShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 100000 --max_expansions 400 --n_processes 15 --output_dir results --output_name 1G_samples
# python monte_carlo_simulation.py --grammar_class PCFGNonDeterministicShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 100000 --max_expansions 400 --n_processes 15 --output_dir results --output_name 1G_samples
# python monte_carlo_simulation.py --grammar_class PCFGLocalShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 100000 --max_expansions 400 --n_processes 15 --output_dir results --output_name 1G_samples
# python monte_carlo_simulation.py --grammar_class PCFGEvenOddShuffle --grammar_file data_gen/base-grammar_eos.gr --n_samples 100000 --max_expansions 400 --n_processes 15 --output_dir results --output_name 1G_samples
