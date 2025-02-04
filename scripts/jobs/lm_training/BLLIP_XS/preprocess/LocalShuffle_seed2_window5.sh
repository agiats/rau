#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name=BLLIP_XS_preprocess_LocalShuffle_seed2_window5
#SBATCH --output=logs/BLLIP_XS/preprocess/LocalShuffle_seed2_window5.out
#SBATCH --error=logs/BLLIP_XS/preprocess/LocalShuffle_seed2_window5.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/BLLIP_XS/LocalShuffle_seed2_window5/train.txt" \
    --validpref "data/fairseq_train/BLLIP_XS/LocalShuffle_seed2_window5/dev.txt" \
    --testpref "data/fairseq_train/BLLIP_XS/LocalShuffle_seed2_window5/test.txt" \
    --destdir "data/fairseq_train_preprocessed/BLLIP_XS/LocalShuffle_seed2_window5" \
    --workers 4
