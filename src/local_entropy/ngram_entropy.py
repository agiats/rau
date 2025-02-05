import gzip
import math
import kenlm
import argparse
import os
import subprocess
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count


def load_data(data_dir: Path, add_eos: bool = True) -> dict:
    """
    Load datasets from the specified directory

    Args:
        data_dir (Path): Path to the data directory
        add_eos (bool): Whether to add [eos] at the end of sentences
    """
    datasets = {}
    all_sentences = []

    # Load each dataset
    for filename in ["train.txt", "dev.txt", "test.txt"]:
        file_path = data_dir / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                sentences = f.read().splitlines()
                if add_eos:
                    sentences = [f"{s} [eos]".strip() for s in sentences]
                dataset_name = filename.split(".")[0]
                datasets[dataset_name] = sentences
                all_sentences.extend(sentences)

    # Add 'all' dataset (train + dev + test)
    datasets["all"] = all_sentences

    return datasets


def calculate_entropy_with_kenlm(model, text, add_eos: bool = True):
    """Calculate the entropy of the text using the KenLM model"""
    need_eos_for_model = not add_eos
    log_prob_sum = 0
    word_count = 0

    for line in text:
        log_prob_sum += model.score(line, eos=need_eos_for_model) * math.log2(10)
        word_count += len(line.split()) + 1

    return -1 * (log_prob_sum / word_count)


def process_single_n(args):
    """
    Function to process a single n-gram model
    """
    work_file_path, n, lmplz_path, datasets, memory, add_eos = args
    arpa_path = work_file_path.with_suffix(f".{n}.arpa")

    try:
        # Train the model with all data (train+dev+test)
        with open(work_file_path, "w") as f:
            for line in datasets["all"]:
                f.write(line + "\n")

        # Create and train the model
        subprocess.run(
            [
                str(lmplz_path),
                "-o",
                str(n),
                "--skip_symbols",
                "--discount_fallback",
                "--memory",
                memory,
                "--text",
                str(work_file_path),
                "--arpa",
                str(arpa_path),
            ],
            check=True,
        )

        model = kenlm.Model(str(arpa_path))

        # Calculate entropy for each dataset
        results = {}
        for dataset_name, sentences in datasets.items():
            results[dataset_name] = calculate_entropy_with_kenlm(
                model, sentences, add_eos=add_eos
            )

        # arpa_path.unlink(missing_ok=True)
        return n, results
    except Exception as e:
        print(f"Error processing {n}-gram: {e}")
        arpa_path.unlink(missing_ok=True)
        return n, None


def main():
    parser = argparse.ArgumentParser(description="Calculate n-gram entropy using KenLM")
    parser.add_argument(
        "input_dir",
        help="Input directory containing train.txt, dev.txt, test.txt",
        type=Path,
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="n-gram sizes (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--kenlm-path", type=str, default="../kenlm", help="Path to KenLM directory"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--memory", type=str, default="8G", help="Memory limit for KenLM (default: 4G)"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work",
        help="Working directory for intermediate files",
    )
    parser.add_argument(
        "--no-eos",
        action="store_false",
        dest="add_eos",
        help="Do not add [eos] token at the end of sentences",
    )
    args = parser.parse_args()

    # Load data
    print("Loading datasets...")
    datasets = load_data(args.input_dir, add_eos=args.add_eos)

    # Create working directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Generate path for the working file
    work_file_path = work_dir / f"{args.input_dir.name}.txt"

    # Set the path to KenLM
    kenlm_build_dir = Path(args.kenlm_path) / "build"
    lmplz_path = kenlm_build_dir / "bin" / "lmplz"

    if not lmplz_path.exists():
        raise FileNotFoundError(f"lmplz not found at {lmplz_path}")

    # Set the number of processes
    num_processes = args.num_processes or cpu_count()
    print(f"Using {num_processes} processes")

    # Execute parallel processing
    process_args = [
        (work_file_path, n, lmplz_path, datasets, args.memory, args.add_eos)
        for n in args.n
    ]
    with Pool(num_processes) as pool:
        results = pool.map(process_single_n, process_args)

    # Convert results to DataFrame
    data = []
    for n, dataset_results in results:
        if dataset_results is not None:
            for dataset_name, entropy in dataset_results.items():
                data.append({"n_gram": n, "dataset": dataset_name, "entropy": entropy})

    df = pd.DataFrame(data)

    # Save as CSV file
    output_file = args.input_dir / "entropy.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Delete the working file
    if work_file_path.exists():
        work_file_path.unlink()


if __name__ == "__main__":
    main()
