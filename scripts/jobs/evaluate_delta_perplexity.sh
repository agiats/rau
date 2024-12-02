#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3g
#SBATCH --tmp=20g
#SBATCH --job-name=evaluate_delta_perplexity
#SBATCH --output=logs/evaluate_delta_perplexity.out
#SBATCH --error=logs/evaluate_delta_perplexity.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate

export PYTHONPATH="."
python scripts/evaluate_delta_perplexity.py
