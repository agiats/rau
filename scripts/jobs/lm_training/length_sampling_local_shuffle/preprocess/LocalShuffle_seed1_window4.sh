#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=length_sampling_local_shuffle_preprocess_LocalShuffle_seed1_window4
#SBATCH --output=logs/length_sampling_local_shuffle/preprocess/LocalShuffle_seed1_window4.out
#SBATCH --error=logs/length_sampling_local_shuffle/preprocess/LocalShuffle_seed1_window4.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/length_sampling_local_shuffle/LocalShuffle_seed1_window4/train.txt" \
    --validpref "data/fairseq_train/length_sampling_local_shuffle/LocalShuffle_seed1_window4/dev.txt" \
    --testpref "data/fairseq_train/length_sampling_local_shuffle/LocalShuffle_seed1_window4/test.txt" \
    --destdir "data/fairseq_train_preprocessed/length_sampling_local_shuffle/LocalShuffle_seed1_window4" \
    --workers 16
