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


def create_splits(input_file, output_dir):
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

    # Calculate split sizes (80-10-10 split)
    train_size = int(0.8 * len(sentences))
    dev_size = int(0.1 * len(sentences))

    # Split the data
    train_data = sentences[:train_size]
    dev_data = sentences[train_size : train_size + dev_size]
    test_data = sentences[train_size + dev_size :]

    logger.info(
        f"Split sizes - Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}"
    )

    # Save splits
    for split_name, data in [
        ("train", train_data),
        ("dev", dev_data),
        ("test", test_data),
    ]:
        output_file = os.path.join(output_dir, f"{split_name}.txt")
        with open(output_file, "w") as f:
            for sentence in data:
                f.write(sentence + "\n")
        logger.info(f"Saved {split_name} data to {output_file}")

    logger.info("Completed splitting data")


def main():
    parser = argparse.ArgumentParser(description="Split data into train/dev/test sets")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input file (can be .gz)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the splits"
    )

    args = parser.parse_args()
    create_splits(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
