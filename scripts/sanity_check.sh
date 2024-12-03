n_samples=100_000_000
exp_dir="results/length_sampling/100M_samples_eos_zipf_min1_max20/true_probs_count_balanced_samples100"


export PYTHONPATH="."
python scripts/sanity_check.py \
    --exp_dir $exp_dir \
    --input_suffix ".csv.gz" \
    --sample_size $n_samples
