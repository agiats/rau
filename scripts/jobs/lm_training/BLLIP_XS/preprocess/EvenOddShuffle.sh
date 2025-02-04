#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name=BLLIP_XS_preprocess_EvenOddShuffle
#SBATCH --output=logs/BLLIP_XS/preprocess/EvenOddShuffle.out
#SBATCH --error=logs/BLLIP_XS/preprocess/EvenOddShuffle.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/BLLIP_XS/EvenOddShuffle/train.txt" \
    --validpref "data/fairseq_train/BLLIP_XS/EvenOddShuffle/dev.txt" \
    --testpref "data/fairseq_train/BLLIP_XS/EvenOddShuffle/test.txt" \
    --destdir "data/fairseq_train_preprocessed/BLLIP_XS/EvenOddShuffle" \
    --workers 4
