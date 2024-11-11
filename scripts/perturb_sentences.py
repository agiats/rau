from hydra.utils import instantiate
import argparse
from pathlib import Path
import gzip
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def process_sentence(perturb_func, sentence):
    return perturb_func.perturb(sentence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=lambda p: Path(p).resolve())
    parser.add_argument("--output_file", type=lambda p: Path(p).resolve())
    parser.add_argument("--perturb_func", type=str, required=True)
    parser.add_argument(
        "--n_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes (default: number of CPU cores)",
    )
    args = parser.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # load sentences
    print("Loading sentences...")
    with gzip.open(args.input_file, "rt", encoding="utf-8") as f:
        sentences = f.read().splitlines()
    # perturb sentences
    perturb_func = instantiate(
        {"_target_": f"src.perturbation.perturbation_func.{args.perturb_func}"}
    )

    print(f"Processing {len(sentences)} sentences with {args.n_workers} workers...")

    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_sentence, perturb_func)
        perturbed_sentences = list(
            tqdm(
                pool.imap(process_func, sentences),
                total=len(sentences),
                desc=f"Perturbing with {args.perturb_func}",
            )
        )
    # save perturbed sentences
    with gzip.open(args.output_file, "wt", encoding="utf-8") as f:
        f.write("\n".join(perturbed_sentences))


if __name__ == "__main__":
    main()
