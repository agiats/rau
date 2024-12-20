import random

def local_k_shuffle(sequence, k, seed=None):
    """
    Local k-shuffle implementation matching LocalShuffle class.
    Preserves random state and uses specified seed.

    Args:
        sequence: Input sequence to shuffle
        k: Block size for local shuffling
        seed: Random seed for reproducibility

    Returns:
        list: Locally shuffled sequence
    """
    state = random.getstate()

    if seed is not None:
        random.seed(seed)

    shuffled_seq = []
    for i in range(0, len(sequence), k):
        batch = sequence[i:min(i + k, len(sequence))].copy()
        random.shuffle(batch)
        shuffled_seq.extend(batch)

    random.setstate(state)

    return shuffled_seq
