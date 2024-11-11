exp_name="100M_samples_eos_zipf_min1_max20"
sample_size=10

python scripts/generate_job_for_true_prob.py \
    --num_jobs 10 \
    --exp_name $exp_name \
    --sample_size $sample_size

