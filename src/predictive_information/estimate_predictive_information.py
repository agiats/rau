import math
import kenlm
import argparse
import json
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional


def load_data(file_path: Path) -> List[str]:
    """
    Load sentences from a single file.
    Each line is treated as a sentence.
    Returns a list of all sentences.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with file_path.open("r") as f:
        sentences = f.read().splitlines()
    
    return sentences


def calculate_mlocal_entropy(model, text, n: int):
    """Calculate n-local entropy (h_n) using KenLM model."""
    total_local_entropy = 0
    denominator = 0

    for line in text:
        scores = list(model.full_scores(line))
        valid_scores = scores[n - 1 :]
        if len(valid_scores) == 0:
            continue
        assert len(valid_scores) == len(line.split()) - n + 2, (
            f"{len(valid_scores)} != {len(line.split()) - n + 2}"
        )
        for prob, _, _ in valid_scores:
            local_entropy = -prob * math.log2(10)
            total_local_entropy += local_entropy
            denominator += 1

    return total_local_entropy / denominator if denominator > 0 else float("inf")


def calculate_predictive_information(
    local_entropies: Dict[int, float], 
    asymptotic_entropy: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate predictive information E = sum_{n=2}^{max_n} (h_n - h_asymptotic)
    
    Args:
        local_entropies: Dictionary mapping n to h_n values
        asymptotic_entropy: The asymptotic entropy rate to use (if None, use last value)
    
    Returns:
        E: Predictive information
        h_asymptotic: The asymptotic entropy rate used
    """
    if not local_entropies:
        return 0.0, 0.0
    
    # Sort by n to ensure we get the last value correctly
    sorted_n = sorted(local_entropies.keys())
    
    # Use provided asymptotic entropy or the last computed value
    if asymptotic_entropy is None:
        h_asymptotic = local_entropies[sorted_n[-1]]
    else:
        h_asymptotic = asymptotic_entropy
    
    # Sum (h_n - h_asymptotic) for all n values
    E = sum(local_entropies[n] - h_asymptotic for n in sorted_n)
    
    return E, h_asymptotic


def check_convergence(
    entropies: Dict[int, float], 
    tol: float = 1e-4, 
    window: int = 3
) -> Optional[int]:
    """
    Check if entropy values have converged.
    
    Args:
        entropies: Dictionary mapping n to h_n values
        tol: Tolerance for convergence
        window: Number of consecutive differences that must be < tol
    
    Returns:
        n value where convergence was detected, or None if not converged
    """
    sorted_n = sorted(entropies.keys())
    
    if len(sorted_n) < window + 1:
        return None
    
    # Check consecutive differences
    for i in range(window, len(sorted_n)):
        differences = []
        for j in range(1, window + 1):
            diff = abs(entropies[sorted_n[i]] - entropies[sorted_n[i-j]])
            differences.append(diff)
        
        if all(diff < tol for diff in differences):
            return sorted_n[i]
    
    return None


def process_single_n(args):
    """Process a single n-gram model and return its local entropy."""
    work_file_path, n, lmplz_path, sentences, memory = args
    arpa_path = work_file_path.with_suffix(f".{n}.arpa")

    try:
        with open(work_file_path, "w") as f:
            for line in sentences:
                f.write(line + "\n")

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
        entropy = calculate_mlocal_entropy(model, sentences, n)
        
        # Clean up ARPA file
        arpa_path.unlink(missing_ok=True)
        
        return n, entropy

    except Exception as e:
        print(f"Error processing {n}-gram: {e}")
        arpa_path.unlink(missing_ok=True)
        return n, None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate predictive information using KenLM n-gram models"
    )
    parser.add_argument(
        "--input_path", 
        type=Path, 
        required=True, 
        help="Path to input text file containing all sentences"
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=10,
        help="Maximum n-gram size (default: 10)"
    )
    parser.add_argument(
        "--min_n",
        type=int,
        default=2,
        help="Minimum n-gram size (default: 2, since h_1 is unconditional entropy)"
    )
    parser.add_argument(
        "--kenlm-path", 
        type=str, 
        default="../kenlm", 
        help="Path to KenLM directory"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--memory", 
        type=str, 
        default="8G", 
        help="Memory limit for KenLM (default: 8G)"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work",
        help="Working directory for intermediate files",
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        required=True, 
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--asymptotic-entropy",
        type=float,
        default=None,
        help="Asymptotic entropy rate to use (if not provided, use last h_n)"
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-4,
        help="Tolerance for convergence detection (default: 1e-4)"
    )
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=3,
        help="Window size for convergence detection (default: 3)"
    )
    args = parser.parse_args()

    print(f"Loading sentences from {args.input_path}...")
    sentences = load_data(args.input_path)
    print(f"Loaded {len(sentences)} sentences")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    work_file_path = work_dir / f"{args.input_path.stem}_work.txt"

    kenlm_build_dir = Path(args.kenlm_path) / "build"
    lmplz_path = kenlm_build_dir / "bin" / "lmplz"

    if not lmplz_path.exists():
        raise FileNotFoundError(f"lmplz not found at {lmplz_path}")

    num_processes = args.num_processes or cpu_count()
    print(f"Using {num_processes} processes")

    # Generate range of n values
    n_values = list(range(args.min_n, args.max_n + 1))
    
    print(f"Calculating entropies for n = {args.min_n} to {args.max_n}...")
    process_args = [
        (work_file_path, n, lmplz_path, sentences, args.memory)
        for n in n_values
    ]
    
    with Pool(num_processes) as pool:
        model_results = pool.map(process_single_n, process_args)

    # Collect entropy results
    local_entropies = {}
    for n, entropy in model_results:
        if entropy is not None:
            local_entropies[n] = entropy
            print(f"h_{n} = {entropy:.6f}")

    if not local_entropies:
        raise ValueError("No entropy values were successfully calculated")

    # Check for convergence
    converged_at = check_convergence(
        local_entropies, 
        tol=args.convergence_tol, 
        window=args.convergence_window
    )
    
    # Calculate predictive information
    E, h_asymptotic = calculate_predictive_information(
        local_entropies, 
        args.asymptotic_entropy
    )
    
    # Prepare results
    final_results = {
        "local_entropies": {str(n): h for n, h in local_entropies.items()},
        "predictive_information": E,
        "h_asymptotic": h_asymptotic,
        "converged_at": converged_at,
        "convergence_tol": args.convergence_tol,
        "convergence_window": args.convergence_window,
        "num_sentences": len(sentences),
        "min_n": args.min_n,
        "max_n": args.max_n
    }
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Number of sentences: {len(sentences)}")
    print(f"h_asymptotic: {h_asymptotic:.6f}")
    if args.asymptotic_entropy is not None:
        print(f"  (provided as argument)")
    else:
        print(f"  (using h_{max(local_entropies.keys())})")
    print(f"Predictive Information E: {E:.6f}")
    if converged_at:
        print(f"Convergence detected at n = {converged_at}")
    else:
        print("No convergence detected within the range")
    print("="*50)

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\nResults saved to {args.output_path}")

    # Clean up
    if work_file_path.exists():
        work_file_path.unlink()


if __name__ == "__main__":
    main()