#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name=ptb_local_shuffle_preprocess_ptb_local_shuffle_LocalShuffle_seed12_window3
#SBATCH --output=logs/ptb_local_shuffle/preprocess/ptb_local_shuffle_LocalShuffle_seed12_window3.out
#SBATCH --error=logs/ptb_local_shuffle/preprocess/ptb_local_shuffle_LocalShuffle_seed12_window3.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed12_window3/train.txt" \
    --validpref "data/fairseq_train/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed12_window3/dev.txt" \
    --testpref "data/fairseq_train/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed12_window3/test.txt" \
    --destdir "data/fairseq_train_preprocessed/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed12_window3" \
    --workers 4
