#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=monte-carlo_100M_samples_expansion_100
#SBATCH --output=monte-carlo_100M_samples_expansion_100.out
#SBATCH --error=monte-carlo_100M_samples_expansion_100.err
#SBATCH --mail-type=END,FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate

n_samples=100_000_000
n_processes=64
output_dir="results"
output_name="100M_samples_expansion_100_zipf"
max_expansions=100
batch_size=50000
grammar_classes=("PCFG" "PCFGDeterministicShuffle" "PCFGNonDeterministicShuffle" "PCFGLocalShuffle" "PCFGEvenOddShuffle" "PCFGNoReverse" "PCFGPartialReverse" "PCFGFullReverse")


# grammar_file="data_gen/base-grammar_eos.gr"
grammar_file="data_gen/base-grammar_eos_zipf.gr"


for grammar_class in "${grammar_classes[@]}"
do
    echo "Grammar class: $grammar_class"
    python monte_carlo_simulation.py \
        --grammar_class $grammar_class \
        --grammar_file $grammar_file \
        --n_samples $n_samples \
        --max_expansions $max_expansions \
        --n_processes $n_processes \
        --output_dir $output_dir \
        --batch_size $batch_size \
        --output_name $output_name
done



# for grammar_class in "${grammar_classes[@]}"
# do
#     echo "Grammar class: $grammar_class"
#     # jq '.Entropy' results/$output_name/$grammar_class/results.json
#     cat results/$output_name/$grammar_class/results.json
# done
