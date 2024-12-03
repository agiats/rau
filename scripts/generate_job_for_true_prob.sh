job_name="100M_samples_eos_zipf_min1_max20_all_samples"
exp_name="100M_samples_eos_zipf_min1_max20"
sample_size=100_000_000

python scripts/generate_job_for_true_prob.py \
    --num_jobs 100 \
    --job_name $job_name \
    --exp_name $exp_name \
    --sample_size $sample_size
