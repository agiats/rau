from hydra.utils import instantiate
import argparse
from pathlib import Path
import gzip
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import json


def process_sentence(perturb_func, sentence):
    return perturb_func.perturb(sentence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=lambda p: Path(p).resolve())
    parser.add_argument("--exp_dir", type=lambda p: Path(p).resolve())
    parser.add_argument("--perturb_config_file", type=lambda p: Path(p).resolve())
    parser.add_argument(
        "--n_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes (default: number of CPU cores)",
    )
    args = parser.parse_args()

    # load sentences
    print("Loading sentences...")
    with gzip.open(args.input_file, "rt", encoding="utf-8") as f:
        sentences = f.read().splitlines()
    # perturb sentences
    with open(args.perturb_config_file, "r") as f:
        perturb_config = json.load(f)

    print(f"Found {len(perturb_config)} perturbation configurations:")
    for k, v in perturb_config.items():
        print(f"{k}: {v}")

    for k, v in perturb_config.items():
        perturb_func = instantiate(v)

        print(f"Processing {len(sentences)} sentences with {args.n_workers} workers...")

        with mp.Pool(processes=args.n_workers) as pool:
            process_func = partial(process_sentence, perturb_func)
            perturbed_sentences = list(
                tqdm(
                    pool.imap(process_func, sentences),
                    total=len(sentences),
                    desc=f"Perturbing with {k}",
                )
            )
        # save perturbed sentences
        output_file = args.exp_dir.parent / f"{args.exp_dir.name}_{k}/samples.txt.gz"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Writing {len(perturbed_sentences)} perturbed sentences to {output_file}..."
        )
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            f.write("\n".join(perturbed_sentences))


if __name__ == "__main__":
    main()
