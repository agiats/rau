from tqdm import tqdm
import multiprocessing as mp

def generate_batch_samples(args):
    """
    Generate multiple samples in a batch to reduce inter-process communication
    """
    model, max_length, batch_size = args
    results = []
    for _ in range(batch_size):
        sequence = model.sample(max_length)
        results.append(sequence)
    return results

def generate_and_save_samples(model, sample_size, max_length, output_file="samples.txt", n_jobs=1):
    """
    Generate multiple samples from the n-gram model and save sequences to a file.
    Each file will contain one sequence per line.
    """
    # Calculate batch size based on sample_size and n_jobs
    batch_size = max(1000, sample_size // (n_jobs * 10))  # 各プロセスが10回程度動くように
    n_batches = (sample_size + batch_size - 1) // batch_size

    # Prepare batched arguments
    args_list = [(model, max_length, min(batch_size, sample_size - i * batch_size))
                 for i in range(n_batches)]

    # Generate samples in batches
    with mp.Pool(n_jobs) as pool:
        all_results = []
        for batch_results in tqdm(
            pool.imap(generate_batch_samples, args_list),
            total=n_batches,
            desc="Generating samples"
        ):
            all_results.extend(batch_results)

    # Save results
    with open(output_file, "w") as f:
        for sequence in all_results:
            f.write(f"{' '.join(sequence)}\n")
