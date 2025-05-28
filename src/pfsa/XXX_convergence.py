from itertools import product
import random
import json
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn style for better aesthetics
sns.set_theme(style="darkgrid")
np.set_printoptions(precision=3)

from fsa_generator import random_dpfsa, random_pfsa, geometric_sum_pfsa, random_ngram

from fsa import PFSA

num_states = 8
num_symbols = 8
A = random_dpfsa(
    num_states,
    num_symbols,
    conditions=[lambda A: 10 < A.mean_length < 80],
    mean_length=20,
    topology_seed=2,
    weight_seed=2,
)
output_dir = Path(f"XXX_convergence_state{num_states}_symbols{num_symbols}")
output_dir.mkdir(parents=True, exist_ok=True)

print("Is A probabilistic?", A.is_probabilistic)
print("length weights sum up to", sum(A.length_p(t) for t in range(100)), "for t=100")

# distribution of length weights
length_weights_distribution = [A.length_p(t) for t in range(1, 100)]
plt.figure(figsize=(10, 6))
sns.histplot(length_weights_distribution, bins=100, kde=True)
plt.title('Distribution of length weights', fontsize=16)
plt.xlabel('Length weight', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig(output_dir / "length_weights_distribution.png", dpi=300)
time_indexed_MI = []
length_weighted_MI = []
length_weights = []

for t in range(1, 10):
    length_weighted_MI.append(A.length_p(t) * A.MI(t))
    time_indexed_MI.append(A.MI(t))
    length_weights.append(A.length_p(t))

# time indexed MI, length weighted MI, length weights
x = np.arange(1, 10)
y = np.array(time_indexed_MI)
y2 = np.array(length_weighted_MI)
y3 = np.array(length_weights)

# Create a better plot with seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x=x, y=y, linewidth=2.5, color='royalblue', label='Time-indexed MI')
sns.lineplot(x=x, y=y2, linewidth=2.5, color='red', label='Length-weighted MI')
sns.lineplot(x=x, y=y3, linewidth=2.5, color='green', label='Length weights')
plt.title('Time-indexed Mutual Information', fontsize=16)
plt.xlabel('Length (t)', fontsize=14)
plt.ylabel('Length_p(t) * MI(t)', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()

# Save with higher DPI for better quality
plt.savefig(output_dir / "time_indexed_mutual_information.png", dpi=300)
print(f"Plot saved to {output_dir / 'time_indexed_mutual_information.png'}")

# local entropies
M = 6
m_local_entropies = []
prefix_m_local_entropies = []
suffix_entropies = []
for m in range(1, M + 1):
    m_local_entropies.append(A.local_entropy(m))
    prefix_m_local_entropy = A.prefix_local_entropy(m)
    prefix_m_local_entropies.append(prefix_m_local_entropy)
    suffix_entropies.append(A.MI(m) + prefix_m_local_entropy)

XXX = A.XXX(T=100)
XXXs = [XXX for _ in range(M)]
global_entropy = A.entropy

# compare XXXs and m_local_entropies and prefix_m_local_entropies and suffix_entropies
# draw entropy as a horizontal line
plt.figure(figsize=(10, 6))
plt.xticks(range(1, M + 1))
sns.lineplot(x=range(1, M + 1), y=XXXs, label='XXX (constant across t)', linewidth=2.5, color='royalblue')
sns.lineplot(x=range(1, M + 1), y=m_local_entropies, label='t-local entropy', linewidth=2.5, color='red')
sns.lineplot(x=range(1, M + 1), y=prefix_m_local_entropies, label='H[Y_{≥t} | Y_{<t}] (prefix local entropy)', linewidth=2.5, color='green')
sns.lineplot(x=range(1, M + 1), y=suffix_entropies, label='H[Y_{≥t}]', linewidth=2.5, color='orange')
sns.lineplot(x=range(1, M + 1), y=[global_entropy for _ in range(M)], label='H[Y] (constant across t)', linewidth=2.5, color='purple')
plt.title('Convergence of XXX and t-local entropy', fontsize=16)
plt.xlabel('t', fontsize=14)
plt.ylabel('Entropy (bits)', fontsize=14)

# save the plot
plt.savefig(output_dir / "XXX_convergence.png", dpi=300)
print(f"Plot saved to {output_dir / 'XXX_convergence.png'}")

# save the actual values
with open(output_dir / "XXX_convergence.json", "w") as f:
    json.dump({
        "XXXs": XXXs,
        "m_local_entropies": m_local_entropies,
        "prefix_m_local_entropies": prefix_m_local_entropies,
        "suffix_entropies": suffix_entropies,
        "time_indexed_MI": time_indexed_MI,
        "length_weighted_MI": length_weighted_MI,
        "length_weights": length_weights,
    }, f)
