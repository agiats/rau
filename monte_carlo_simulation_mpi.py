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
from mpi4py import MPI

class PCFG:
    """
    PCFG to sample sentences from
    """
    def __init__(self, grammar_file):
        self.rules = None
        self.change_rules = None
        self.load_rules(grammar_file)

    def load_rules(self, grammar_file):
        new_rules = {}
        change = {}
        g_file = open(grammar_file, 'r')
        lines = g_file.readlines()
        for l in lines:
            if l.startswith(('#', " ", "\t", "\n")) or len(l) < 1:
                continue
            else:
                if l.find("#") != -1:
                    l = l[:l.find("#")]
                idx = -1
                if len(l.rstrip().split("\t")) == 3:
                    weight, lhs, rhs = l.rstrip().split("\t")
                elif len(l.rstrip().split("\t")) == 4:
                    weight, lhs, rhs, idx = l.rstrip().split("\t")
                if lhs not in new_rules.keys():
                    new_rules[lhs] = []
                poss_rhs = new_rules[lhs]
                poss_rhs.append([rhs, float(weight)])
                if idx != -1:
                    change[lhs + "\t" + rhs] = idx
        for lhs, poss in new_rules.items():
            total = 0
            for rhs in poss:
                total += rhs[1]
            for rhs in poss:
                rhs[1] /= total
        self.rules = new_rules
        self.change_rules = change

    def sample_sentence(self, max_expansions, bracketing):
        self.expansions = 0
        done = False
        sent = ["ROOT"]
        idx = 0
        while not done:
            if sent[idx] not in self.rules.keys():
                idx += 1
                if idx >= len(sent):
                    done = True
                continue
            else:
                replace, change_idx = self.expand(sent[idx])
                if bracketing:
                    if change_idx == -1:
                        sent = (sent[:idx]
                            + ["(", sent[idx]] + replace + [")"]
                            + sent[idx + 1:])
                    else:
                        sent = (sent[:idx]
                            + ["(", change_idx + sent[idx]] + replace + [")"]
                            + sent[idx + 1:])
                else:
                    sent = sent[:idx] + replace  + sent[idx + 1:]
                self.expansions += 1
                if bracketing:
                    idx += 2
                if self.expansions > max_expansions:
                    done = True
                if idx >= len(sent):
                    done = True
        if self.expansions > max_expansions:
            # print("Max expansions reached")
            return None
            # for idx in range(len(sent)):
            #     if not bracketing:
            #         if sent[idx] in self.rules.keys():
            #             sent[idx] = "..."
            #     else:
            #         if sent[idx] in self.rules.keys() and sent[idx - 1] != "(":
            #             sent[idx] = "..."
        return ' '.join(sent)

    def expand(self, symbol):
        poss = self.rules[symbol]
        sample = random.random()
        val = 0.0
        rhs = ""
        idx = -1
        for p in poss:
            val += p[1]
            if sample <= val:
                if symbol + "\t" + p[0] in self.change_rules.keys():
                    idx = self.change_rules[symbol + "\t" + p[0]]
                rhs = p[0]
                break
        return rhs.split(" "), idx


def sample_batch(args):
    grammar_file, max_expansions, batch_size, process_id = args
    grammar = PCFG(grammar_file)
    sentence_counts = Counter()
    successful_samples = 0

    # バッチ内でサンプリング
    for _ in range(batch_size):
        sent = grammar.sample_sentence(max_expansions=max_expansions, bracketing=False)
        if sent is not None:
            sentence_counts[sent] += 1
            successful_samples += 1

    return process_id, successful_samples, sentence_counts

def monte_carlo_parallel(grammar_file, max_expansions, n_samples, batch_size=10000, n_processes=None):
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
                        batch_args.append((grammar_file, max_expansions, process_batch_size, i))

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

    sentence_probs = {k: v/total_samples for k, v in total_counts.items()}
    return total_counts, sentence_probs, total_samples

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Parallel Monte Carlo sampling for PCFG")
    parser.add_argument("grammar_file", type=str, help="Path to grammar file")
    parser.add_argument("--n_samples", type=int, default=1_000_000, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Sub-batch size")
    parser.add_argument("--max_expansions", type=int, default=100, help="Max number of expansions")
    parser.add_argument("--n_processes", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--output_name", type=str, default="samples", help="Output file name")

    args = parser.parse_args()

    samples_per_process = args.n_samples // size
    if rank < args.n_samples % size:
        samples_per_process += 1

    # 出力ディレクトリにプロセスランクを追加
    output_dir = Path(f"{args.output_dir}/{args.output_name}")
    output_dir.mkdir(exist_ok=True, parents=True)

    # モンテカルロサンプリングの実行
    counts, probs, actual_samples = monte_carlo_parallel(
        args.grammar_file,
        args.max_expansions,
        samples_per_process,  # プロセスごとのサンプル数
        args.batch_size,
        args.n_processes
    )


    # 結果の保存
    with open(output_dir / f"sentence_counts_rank{rank}.json", "w") as f:
        json.dump(counts, f, indent=2)

    with open(output_dir /f"sentence_probs_rank{rank}.json", "w") as f:
        json.dump(probs, f, indent=2)

    print(f"\nRequested samples: {args.n_samples}")
    print(f"Actual samples: {actual_samples}")
    print(f"Total unique sentences: {len(counts)}")
    print(f"Most common sentence: {max(counts.items(), key=lambda x: x[1])}")
    print(f"Least common sentence: {min(counts.items(), key=lambda x: x[1])}")

if __name__ == "__main__":
    main()
# python monte_carlo_simulation.py data_gen/base-grammar_eos.gr --n_samples 1_000_000_000 --max_expansions 400 --n_processes 15 --output_dir results --output_name 1G_samples --batch_size 100000
