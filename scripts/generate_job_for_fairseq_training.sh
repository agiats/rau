python scripts/generate_job_for_fairseq_training.py \
    --grammar_names "Base" "DeterministicShuffle" "NonDeterministicShuffle" "LocalShuffle" "EvenOddShuffle" "NoReverse" "PartialReverse" "FullReverse" \
    --output_dir scripts/jobs/lm_training


# for f in $(find scripts/jobs/lm_training/lm_transformer_100K_dedup -path '*/lm_*' -name '*.sh'); do sbatch $f; done
