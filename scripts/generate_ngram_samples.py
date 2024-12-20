import string
import argparse
from src.local_entropy.ngram_model import NGramModel
from src.local_entropy.generator import generate_and_save_samples

def main():
    parser = argparse.ArgumentParser(description='Generate samples using n-gram model')
    parser.add_argument('--n', type=int, default=4, help='n-gram size')
    parser.add_argument('--alpha', type=float, default=1.0, help='Dirichlet concentration parameter')
    parser.add_argument('--sample-size', type=int, default=1000000, help='Number of samples to generate')
    parser.add_argument('--max-length', type=int, default=-1, help='Maximum sequence length (-1 for unlimited)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel processes')
    parser.add_argument('--output-file', default='samples.txt', help='Path to save generated sequences')
    parser.add_argument('--save-model', type=str, help='Path to save the model')
    parser.add_argument('--load-model', type=str, help='Path to load existing model')

    args = parser.parse_args()

    # Setup model
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = NGramModel.load(args.load_model)
    else:
        print("Creating new model")
        base_alphabet = list(string.ascii_lowercase)[:20]
        alphabet = ['[BOS]', '[EOS]'] + base_alphabet
        model = NGramModel(alphabet=alphabet, n=args.n, alpha=args.alpha)

        if args.save_model:
            print(f"Saving model to {args.save_model}")
            model.save(args.save_model)

    # Generate samples
    generate_and_save_samples(
        model=model,
        sample_size=args.sample_size,
        max_length=args.max_length,
        output_file=args.output_file,
        n_jobs=args.n_jobs
    )

if __name__ == "__main__":
    main()
