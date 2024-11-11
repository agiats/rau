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


def sample_batch(
    sampler, generator_class, valid_lengths, batch_size, seed, count_only=False
):
    local_generator = generator_class(seed)
    if count_only:
        counter = Counter()
        pbar = tqdm(total=batch_size, desc=f"Counting batch", leave=False)
        for _ in range(batch_size):
            length = local_generator.choice(valid_lengths)
            sentence = " ".join(sampler.sample(length, local_generator))
            counter[sentence] += 1
            pbar.update(1)
        pbar.close()
        return counter
    else:
        local_samples = []
        pbar = tqdm(total=batch_size, desc=f"Sampling batch", leave=False)
        while len(local_samples) < batch_size:
            length = local_generator.choice(valid_lengths)
            local_samples.append(list(sampler.sample(length, local_generator)))
            pbar.update(1)
        pbar.close()
        return local_samples


def sample_with_length_constraint(
    sampler, generator, valid_lengths, sample_size, num_workers=1, count_only=False
):
    print(
        f"{'Counting' if count_only else 'Sampling'} {sample_size} sequences using {num_workers} workers..."
    )

    if num_workers <= 1:
        return sample_batch(
            sampler,
            generator.__class__,
            valid_lengths,
            sample_size,
            generator.randint(0, 2**32),
            count_only,
        )

    # Calculate batch size for each worker
    batch_size = (sample_size + num_workers - 1) // num_workers
    batches = [
        min(batch_size, sample_size - i * batch_size) for i in range(num_workers)
    ]
    seeds = [generator.randint(0, 2**32) for _ in range(num_workers)]

    # Execute sampling in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                sample_batch,
                sampler,
                generator.__class__,
                valid_lengths,
                batch,
                seed,
                count_only,
            )
            for batch, seed in zip(batches, seeds)
        ]

        if count_only:
            total_counter = Counter()
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing batches",
            ):
                total_counter.update(future.result())
            return total_counter
        else:
            all_samples = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing batches",
            ):
                all_samples.extend(future.result())
            return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_file", type=str)
    parser.add_argument("--start_symbol", type=str, default="S")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_path", type=lambda p: Path(p).resolve())
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel sampling",
    )
    parser.add_argument(
        "--output_sent",
        action="store_true",
        default=False,
        help="If True, output sentences. If False, output only counts.",
    )
    args = parser.parse_args()
    generator, seed = get_random_generator_and_seed(args.seed)
    grammar = Grammar.from_file(
        args.grammar_file, Nonterminal(args.start_symbol), args.normalize
    )
    sampler = construct_pcfg_sampler(grammar)
    valid_lengths = sampler.valid_lengths(args.min_length, args.max_length)

    print("Valid lengths:", valid_lengths)
    print("Sampling...")
    samples = sample_with_length_constraint(
        sampler,
        generator,
        valid_lengths,
        args.num_samples,
        args.num_workers,
        count_only=not args.output_sent,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_sent:
        print(f"Writing {len(samples)} sentences to {args.output_path}...")
        with gzip.open(args.output_path, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(" ".join(sample) + "\n")
    else:
        total_samples = sum(samples.values())
        unique_samples = len(samples)
        min_count = min(samples.values())
        max_count = max(samples.values())

        print(f"\nSampling Statistics:")
        print(f"Total samples: {total_samples}")
        print(
            f"Unique samples: {unique_samples} ({unique_samples/total_samples*100:.2f}%)"
        )
        print(f"Min count: {min_count}")
        print(f"Max count: {max_count}")

        print(f"\nWriting counts to {args.output_path}...")
        with gzip.open(args.output_path, "wt", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
    # Save valid lengths
    with open(args.output_path.parent / "valid_lengths.txt", "w") as f:
        f.write(",".join(map(str, valid_lengths)))


if __name__ == "__main__":
    main()
