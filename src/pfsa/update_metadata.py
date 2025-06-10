import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from fsa import PFSA


def calculate_predictive_information(local_entropies: Dict[str, float], asymptotic_entropy: float) -> float:
    """
    Calculate predictive information E = sum_{n=2}^{max_n} (h_n - h_asymptotic)

    Args:
        local_entropies: Dictionary mapping m (as string) to local_entropy(m) values
        asymptotic_entropy: The asymptotic entropy rate to use

    Returns:
        E: Predictive information
    """
    if not local_entropies:
        return 0.0

    # Sum (h_n - h_asymptotic) for all available n values
    E = sum(h_n - asymptotic_entropy for h_n in local_entropies.values())
    return E


def calculate_pfsa_metadata(
    pfsa: PFSA,
    existing_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate metadata from a PFSA object.

    Args:
        pfsa: PFSA object
        existing_metadata: Existing metadata to use for some calculations

    Returns:
        Dictionary with calculated metadata
    """
    metadata = {}
    existing = existing_metadata or {}

    # Local entropies - can reuse existing ones or calculate new ones
    if 'local_entropy' in existing:
        local_entropies = existing['local_entropy']
        metadata['local_entropy'] = local_entropies
    else:
        print("Calculating local entropies...")
        local_entropies = {}
        for m in range(2, 11):  # Calculate up to 10-local entropy
            local_entropies[str(m)] = float(pfsa.local_entropy(m))
    metadata['local_entropy'] = local_entropies

    # Predictive information
    if local_entropies:
        # 1. Using the empirical convergence value (last local_entropy)
        sorted_m = sorted(local_entropies.keys(), key=int)
        h_asymptotic_empirical = local_entropies[sorted_m[-1]]
        E_empirical = calculate_predictive_information(local_entropies, h_asymptotic_empirical)

        # 2. Using next_symbol_entropy (stationary entropy rate)
        E_stationary = calculate_predictive_information(local_entropies, metadata['next_symbol_entropy'])

        metadata['predictive_information_empirical'] = E_empirical
        metadata['predictive_information_stationary'] = E_stationary
        metadata['h_asymptotic_empirical'] = h_asymptotic_empirical

    return metadata


def update_metadata(
    model_path: Path,
    metadata_path: Path,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Update metadata with newly calculated values from a PFSA model.

    Args:
        model_path: Path to the PFSA pickle file
        metadata_path: Path to the existing metadata JSON file
        output_path: Path to save updated metadata (if None, overwrites original)

    Returns:
        Updated metadata dictionary
    """
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        existing_metadata = json.load(f)

    # Load PFSA model
    print(f"Loading PFSA model from {model_path}...")
    pfsa = PFSA(fname=str(model_path))

    # Calculate all metadata
    print(f"Calculating metadata...")
    new_metadata = calculate_pfsa_metadata(pfsa, existing_metadata)

    # Update existing metadata with new values
    metadata = existing_metadata.copy()
    metadata.update(new_metadata)

    # Save updated metadata
    output_path = output_path or metadata_path
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Updated metadata saved to {output_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Update PFSA metadata with newly calculated values"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to the PFSA pickle file"
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        required=True,
        help="Path to the existing metadata JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Path to save updated metadata (if not provided, overwrites original)"
    )

    args = parser.parse_args()

    update_metadata(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
