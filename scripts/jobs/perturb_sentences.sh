#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=perturb_sentences
#SBATCH --output=perturb_sentences.out
#SBATCH --error=perturb_sentences.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


input_file="/cluster/home/tsomeya/projects/impossible_inherent_entropy/results/length_sampling/100M_samples_eos_zipf_min1_max20/samples.txt.gz"
perturb_funcs=("NoReverse" "PartialReverse" "FullReverse" "DeterministicShuffle" "NonDeterministicShuffle" "LocalShuffle" "EvenOddShuffle")
for perturb_func in "${perturb_funcs[@]}"; do
    output_file=$(dirname "$input_file")_${perturb_func}/$(basename "$input_file" .gz)
    echo "Perturbing $input_file with $perturb_func and saving to $output_file"
    python scripts/perturb_sentences.py \
        --input_file "$input_file" \
        --output_file "$output_file" \
        --perturb_func "$perturb_func"
done
