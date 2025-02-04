#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name=local_entropy_preprocess_Q16_S64_s14946
#SBATCH --output=logs/local_entropy/preprocess/Q16_S64_s14946.out
#SBATCH --error=logs/local_entropy/preprocess/Q16_S64_s14946.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/local_entropy/Q16_S64_s14946/train.txt" \
    --validpref "data/fairseq_train/local_entropy/Q16_S64_s14946/dev.txt" \
    --testpref "data/fairseq_train/local_entropy/Q16_S64_s14946/test.txt" \
    --destdir "data/fairseq_train_preprocessed/local_entropy/Q16_S64_s14946" \
    --workers 4
