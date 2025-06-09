"""
Calculate predictive information (excess entropy) for PFSA.

Predictive information E = I[X<t : X≥t] measures the mutual information
between the past and future of a stochastic process.

It can be calculated as E = Σ_{n=2}^∞ (h_n - h)
where h_n is the n-gram entropy rate (starting from n=2) and h is the asymptotic entropy rate.

Important Note:
The PFSA's local_entropy method uses infix probabilities (averaging over all occurrences
of contexts within strings), not prefix probabilities (conditioning on the beginning of
strings). This can lead to different convergence behavior than expected. The asymptotic
rate for infix-based calculations may differ from next_symbol_entropy (the stationary
entropy rate).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from tqdm import trange  # noqa: F401


def calculate_predictive_information(pfsa, max_n: int = 20) -> Tuple[float, List[float], float]:
    """
    Calculate predictive information E = sum_{n=2}^{inf} (h_n - h)
    where h_n is the n-gram entropy rate and h is the asymptotic entropy rate.

    Note: We start from n=2 since h_1 would be unconditional entropy.
    h_n for n>=2 represents H[X_n | X_1, ..., X_{n-1}]

    Args:
        pfsa: PFSA object
        max_n: Maximum n-gram order to compute

    Returns:
        E: Predictive information
        h_values: List of n-gram entropy rates (starting from h_2)
        h_asymptotic: Asymptotic entropy rate (last computed h_n value)
    """
    h_values = []

    # Calculate n-gram entropy rates starting from n=2
    for n in range(2, max_n + 1):
        h_n = pfsa.local_entropy(m=n)
        h_values.append(h_n)

    # The asymptotic entropy rate is simply the last computed value
    # This represents our best estimate of the limit
    h_asymptotic = h_values[-1] if h_values else 0.0

    # Calculate predictive information
    E = sum(h_n - h_asymptotic for h_n in h_values)

    return E, h_values, h_asymptotic


def calculate_predictive_information_with_convergence(
    pfsa,
    max_n: int = 50,
    tol: float = 1e-6,
    window: int = 3
) -> Tuple[float, List[float], float, Optional[int]]:
    """
    Calculate predictive information with automatic convergence detection.

    Note: We start from n=2 since h_1 would be unconditional entropy.

    Important: The PFSA's local_entropy method uses infix probabilities, not prefix
    probabilities. This means the asymptotic rate is the empirical limit of the
    local_entropy values, which may differ from next_symbol_entropy (the stationary
    entropy rate). For true predictive information, prefix-based calculations would
    be more appropriate.

    Args:
        pfsa: PFSA object
        max_n: Maximum n-gram order to compute
        tol: Tolerance for convergence (max acceptable change between consecutive h_n)
        window: Number of consecutive differences that must be < tol for convergence

    Returns:
        E: Predictive information
        h_values: List of n-gram entropy rates (starting from h_2)
        h_asymptotic: Asymptotic entropy rate (last computed h_n value)
        converged_at: n value where convergence was detected
    """
    h_values = []
    converged_at = None

    for n in range(2, max_n + 1):
        h_n = pfsa.local_entropy(m=n)
        h_values.append(h_n)

        # Check for convergence: successive differences are decreasing and small
        if len(h_values) >= window:
            # Check if all recent consecutive differences are small
            differences = [abs(h_values[i] - h_values[i-1])
                          for i in range(-window+1, 0)]
            if all(diff < tol for diff in differences):
                converged_at = n
                break

    # The asymptotic entropy rate is simply the last computed value
    h_asymptotic = h_values[-1] if h_values else 0.0

    # Calculate predictive information
    E = sum(h_n - h_asymptotic for h_n in h_values)

    return E, h_values, h_asymptotic, converged_at


def plot_entropy_convergence(
    h_values: List[float],
    h_asymptotic: float,
    E: float,
    converged_at: Optional[int] = None,
    title: str = "Convergence of n-gram Entropy Rates",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Visualize the convergence of n-gram entropy rates to asymptotic entropy rate.

    Args:
        h_values: List of n-gram entropy rates
        h_asymptotic: Asymptotic entropy rate
        E: Predictive information
        converged_at: n value where convergence was detected (optional)
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    n_vals = range(2, len(h_values) + 2)  # Start from n=2

    # Plot h_n values
    plt.plot(n_vals, h_values, 'b-o', label='$h_n$ (n-gram entropy rate)', markersize=6)

    # Plot asymptotic entropy rate
    plt.axhline(y=h_asymptotic, color='r', linestyle='--', linewidth=2,
                label=f'$h$ (asymptotic) = {h_asymptotic:.4f}')

    # Shade the area representing predictive information
    plt.fill_between(n_vals, h_asymptotic, h_values, alpha=0.3, color='green',
                     label=f'Predictive Information E = {E:.4f}')

    # Mark convergence point if provided
    if converged_at:
        plt.axvline(x=converged_at, color='orange', linestyle=':', linewidth=2,
                    label=f'Converged at n={converged_at}')

    plt.xlabel('n (context length)', fontsize=12)
    plt.ylabel('Entropy rate (bits)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_predictive_information_across_parameters(
    n_states_list: List[int],
    n_symbols_list: List[int],
    n_samples: int = 10,
    max_n: int = 15,
    mean_length: float = 20,
    topology_seed: int = 2,
    plot_heatmap: bool = True
) -> List[dict]:
    """
    Analyze predictive information across different PFSA configurations.

    Args:
        n_states_list: List of state counts to test
        n_symbols_list: List of symbol counts to test
        n_samples: Number of random samples per configuration
        max_n: Maximum n-gram order for entropy calculation
        mean_length: Target mean length for generated sequences
        topology_seed: Seed for topology generation
        plot_heatmap: Whether to plot results as heatmap

    Returns:
        List of results dictionaries
    """
    from src.pfsa.fsa_generator import random_dpfsa

    results = []

    for n_states in n_states_list:
        for n_symbols in n_symbols_list:
            E_values = []
            h_asymptotic_values = []

            for _ in range(n_samples):
                # Generate random PFSA
                pfsa = random_dpfsa(
                    n_states,
                    n_symbols,
                    conditions=[lambda A: 10 < A.mean_length < 30],
                    mean_length=mean_length,
                    topology_seed=topology_seed,
                    weight_seed=np.random.randint(0, 10000),
                )

                # Calculate predictive information
                E, _, h_asymptotic, _ = calculate_predictive_information_with_convergence(
                    pfsa, max_n=max_n
                )

                E_values.append(E)
                h_asymptotic_values.append(h_asymptotic)

            results.append({
                'n_states': n_states,
                'n_symbols': n_symbols,
                'predictive_info_mean': np.mean(E_values),
                'predictive_info_std': np.std(E_values),
                'entropy_rate_mean': np.mean(h_asymptotic_values),
                'entropy_rate_std': np.std(h_asymptotic_values),
            })

    if plot_heatmap:
        # Create heatmap of mean predictive information
        pivot_data = np.zeros((len(n_states_list), len(n_symbols_list)))
        for r in results:
            i = n_states_list.index(r['n_states'])
            j = n_symbols_list.index(r['n_symbols'])
            pivot_data[i, j] = r['predictive_info_mean']

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_data,
                    xticklabels=n_symbols_list,
                    yticklabels=n_states_list,
                    annot=True,
                    fmt='.3f',
                    cmap='viridis')
        plt.xlabel('Number of Symbols')
        plt.ylabel('Number of States')
        plt.title(f'Mean Predictive Information (n={n_samples} samples per config)')
        plt.tight_layout()
        plt.show()

    return results


def plot_entropy_rate_comparison(
    pfsa_list: List,
    labels: List[str],
    max_n: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Compare entropy rate convergence across multiple PFSAs.

    Args:
        pfsa_list: List of PFSA objects to compare
        labels: List of labels for each PFSA
        max_n: Maximum n-gram order to compute
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(pfsa_list)))

    for i, (pfsa, label) in enumerate(zip(pfsa_list, labels)):
        E, h_values, h_asymptotic, converged_at = calculate_predictive_information_with_convergence(
            pfsa, max_n=max_n
        )

        n_vals = range(2, len(h_values) + 2)  # Start from n=2
        plt.plot(n_vals, h_values, '-o', color=colors[i],
                 label=f'{label} (E={E:.3f}, h={h_asymptotic:.3f})',
                 markersize=4, alpha=0.8)

        # Add asymptotic line
        plt.axhline(y=h_asymptotic, color=colors[i], linestyle='--',
                    alpha=0.5, linewidth=1)

    plt.xlabel('n (context length)', fontsize=12)
    plt.ylabel('Entropy rate h_n (bits)', fontsize=12)
    plt.title('Entropy Rate Convergence Comparison', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Example usage
    from src.pfsa.fsa_generator import random_dpfsa

    # Generate a sample PFSA
    pfsa = random_dpfsa(
        n_states=3,
        n_symbols=3,
        conditions=[lambda A: 10 < A.mean_length < 80],
        mean_length=20,
        topology_seed=2,
        weight_seed=2,
    )

    # Calculate predictive information
    E, h_values, h_asymptotic, converged_at = calculate_predictive_information_with_convergence(
        pfsa, max_n=15, tol=1e-5, window=3
    )

    print(f"Predictive Information E = {E:.4f}")
    print(f"Asymptotic entropy rate h = {h_asymptotic:.4f}")
    if converged_at:
        print(f"Converged at n = {converged_at}")
    else:
        print("Did not converge within max_n iterations")

    # Plot convergence
    plot_entropy_convergence(h_values, h_asymptotic, E, converged_at)
