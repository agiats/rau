#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name=test_preprocess
#SBATCH --output=logs/test_preprocess.out
#SBATCH --error=logs/test_preprocess.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.
　
# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/test/train.txt" \
    --validpref "data/test/dev.txt" \
    --testpref "data/test/test.txt" \
    --destdir "data/test_processed" \
    --workers 4
