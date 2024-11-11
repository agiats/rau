n_samples=100_000_000
min_length=1
max_length=20
exp_dir="results/length_sampling/100M_samples_eos_zipf_min${min_length}_max${max_length}"


export PYTHONPATH="."
python scripts/sanity_check.py \
    --exp_dir $exp_dir \
    --input_suffix ".json.gz" \
    --sample_size $n_samples
