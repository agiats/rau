import argparse
import glob
import json
import os
import sys
from pathlib import Path

def extract_sentences_from_babylm(file_pattern, min_length=2):
    """
    Extract sentences from BabyLM JSON files.

    Args:
        file_pattern (str): Pattern to match JSON files
        min_length (int): Minimum number of characters for a sentence

    Returns:
        list: List of extracted sentences
    """
    all_sentences = []

    # Get all matching files
    json_files = glob.glob(file_pattern)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found matching pattern: {file_pattern}")

    for file_path in json_files:
        try:
            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract sentences from each annotation
            for item in data:
                for annotation in item['sent_annotations']:
                    sent = annotation['sent_text']
                    # Only include sentences that meet minimum length requirement
                    if len(sent.strip()) >= min_length:
                        all_sentences.append(sent)

        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON file: {file_path}", file=sys.stderr)
        except KeyError as e:
            print(f"Error: Missing key in JSON structure: {e} in file: {file_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}", file=sys.stderr)

    return all_sentences

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert BabyLM JSON files to a single text file')
    parser.add_argument('input_dir', help='Directory containing JSON files')
    parser.add_argument('output_path', help='Path to output text file')
    parser.add_argument('--min-length', type=int, default=2,
                      help='Minimum number of characters for a sentence (default: 2)')

    args = parser.parse_args()

    try:
        # Verify input directory exists
        if not os.path.isdir(args.input_dir):
            raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")

        # Create file pattern from input directory
        file_pattern = os.path.join(args.input_dir, "*.json")

        # Extract all sentences
        sentences = extract_sentences_from_babylm(file_pattern, args.min_length)

        if not sentences:
            print("Warning: No sentences were extracted", file=sys.stderr)
            return

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Write sentences to output file
        with open(args.output_path, 'w') as f:
            for sentence in sentences:
                f.write(sentence + '\n')

        print(f"Processed {len(sentences)} sentences")
        print(f"Output written to: {args.output_path}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
