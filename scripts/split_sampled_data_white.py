import os
import gzip
import argparse
import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def create_splits(input_file, output_dir, num_splits, num_samples_per_split):
    logger = setup_logger()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Read input file
    logger.info(f"Reading input file: {input_file}")
    if input_file.endswith(".gz"):
        with gzip.open(input_file, "rt") as f:
            sentences = [line.strip() for line in f]
    else:
        with open(input_file, "r") as f:
            sentences = [line.strip() for line in f]
    logger.info(f"Loaded {len(sentences)} sentences")

    # remove [eos] at the end
    sentences = [" ".join(sentence.split()[:-1]) for sentence in sentences]

    # Process each split
    for split_idx in range(num_splits):
        logger.info(f"Processing split {split_idx + 1}/{num_splits}")

        # Take num_samples_per_split sentences from the beginning
        start_idx = split_idx * num_samples_per_split
        split_sentences = sentences[start_idx : start_idx + num_samples_per_split]

        if not split_sentences:
            logger.warning(f"No more sentences available for split {split_idx}")
            break

        logger.info(f"Got {len(split_sentences)} sentences for split {split_idx}")

        # Calculate split sizes (80-10-10 split)
        train_size = int(0.8 * len(split_sentences))
        dev_size = int(0.1 * len(split_sentences))

        # Split the data
        train_data = split_sentences[:train_size]
        dev_data = split_sentences[train_size : train_size + dev_size]
        test_data = split_sentences[train_size + dev_size :]

        logger.info(
            f"Split sizes - Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}"
        )

        # Save splits
        for split_name, data in [
            ("train", train_data),
            ("dev", dev_data),
            ("test", test_data),
        ]:
            output_file = os.path.join(output_dir, f"split_{split_idx}.{split_name}")
            with open(output_file, "w") as f:
                for sentence in data:
                    f.write(sentence + "\n")
            logger.info(f"Saved {split_name} data to {output_file}")

    logger.info("Completed all splits")


def main():
    parser = argparse.ArgumentParser(
        description="Split sampled data into train/dev/test sets"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input file (can be .gz)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the splits"
    )
    parser.add_argument(
        "--num_splits", type=int, default=10, help="Number of splits to create"
    )
    parser.add_argument(
        "--num_samples_per_split",
        type=int,
        default=10000,
        help="Number of samples per split",
    )

    args = parser.parse_args()
    create_splits(
        args.input_file, args.output_dir, args.num_splits, args.num_samples_per_split
    )


if __name__ == "__main__":
    main()
