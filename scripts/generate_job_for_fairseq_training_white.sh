python scripts/generate_job_for_fairseq_training_white.py \
    --grammar_names "Base" "DeterministicShuffle" "NonDeterministicShuffle" "LocalShuffle" "EvenOddShuffle" "NoReverse" "PartialReverse" "FullReverse" \
    --num_splits 10 \
    --output_dir scripts/jobs/lm_training
