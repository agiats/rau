import argparse
from pathlib import Path
import json

import torch
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.vocabulary import (
    load_vocabulary_data_from_file,
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Initialize and save a random LM with a dummy vocabulary.'
        )
    )
    p.add_argument(
        '--vocabulary-file',
        type=Path,
        required=True,
        help='Path to a .vocab file defining the vocabulary.',
    )
    # ModelInterface will add: --output, --parameter-seed and architecture args
    return p


def main() -> None:
    parser = make_parser()
    # Use ModelInterface to inject architecture arguments
    model_interface = LanguageModelingModelInterface(use_init=True)
    model_interface.add_arguments(parser)
    args = parser.parse_args()

    # Load vocabulary data to determine vocab sizes
    vocab_data = load_vocabulary_data_from_file(args.vocabulary_file)

    # Construct and save an initialized model
    saver = model_interface.construct_saver(args, vocab_data)
    # Save randomly initialized parameters immediately (compatible with
    # older RAU)
    try:
        save_params = getattr(saver, 'save_parameters')
    except AttributeError:
        save_params = None
    if callable(save_params):
        save_params()
    else:
        # Fallback: write parameters.pt directly
        params_path = getattr(saver, 'parameters_file', None)
        if params_path is None:
            params_path = Path(args.output) / 'parameters.pt'
        torch.save(saver.model.state_dict(), params_path)
    # Ensure kwargs.json exists
    kwargs_path = getattr(saver, 'kwargs_file', None)
    if kwargs_path is None:
        kwargs_path = Path(args.output) / 'kwargs.json'
    if not Path(kwargs_path).exists():
        try:
            data = getattr(saver, 'kwargs')
        except Exception:
            data = {}
        with open(kwargs_path, 'w') as fout:
            json.dump(data, fout, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()

