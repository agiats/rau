#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=length_sampling_100M_pcfg_eos_zipf
#SBATCH --output=length_sampling_100M_pcfg_eos_zipf.out
#SBATCH --error=length_sampling_100M_pcfg_eos_zipf.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate

n_samples=100
n_processes=64
grammar_name="base-grammar_eos_zipf.gr"
min_length=1
max_length=20
output_name="100_samples_eos_zipf_min${min_length}_max${max_length}"


export PYTHONPATH="."
python scripts/length_sampling.py \
    --grammar_file data_gen/${grammar_name} \
    --start_symbol "S" \
    --normalize \
    --seed 42 \
    --min_length $min_length \
    --max_length $max_length \
    --num_samples $n_samples \
    --output_path "results/length_sampling/${output_name}/samples.txt.gz" \
     --num_workers $n_processes
