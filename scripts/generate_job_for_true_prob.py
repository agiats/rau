import json
import math
import argparse
from pathlib import Path


def generate_job_script(
    job_id: int,
    start_idx: int,
    end_idx: int,
    total_jobs: int,
    exp_name: str,
) -> str:
    return f"""#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name={exp_name}_100M_samples_split_{job_id}_of_{total_jobs}
#SBATCH --output=logs/{exp_name}_100M_samples/split_{job_id}_of_{total_jobs}.out
#SBATCH --error=logs/{exp_name}_100M_samples/split_{job_id}_of_{total_jobs}.err

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

grammar_name="base-grammar_eos_zipf.gr"
n_processes=128

python scripts/calculate_true_prob.py \\
    --grammar_file data_gen/$grammar_name \\
    --start_symbol "S" \\
    --normalize \\
    --sentence_counts_path "results/length_sampling/{exp_name}/sample_counts.csv.gz" \\
    --output_path "results/length_sampling/{exp_name}/true_probs_100M_samples/probability_split_{job_id}_of_{total_jobs}.csv.gz" \\
    --num_workers $n_processes \\
    --start_index {start_idx} \\
    --end_index {end_idx}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate split job scripts for sentence processing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scripts/jobs",
        help="Directory to store the generated job scripts",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=10,
        help="Number of jobs to split into (default: 10)",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        required=True,
        help="Name of the job scripts (default: count_balanced_samples100)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (e.g., 100M_samples_expansion_30_zipf)",
    )
    parser.add_argument(
        "--sample_size", type=int, required=True, help="Number of sentences to sample"
    )
    args = parser.parse_args()

    sentences_per_job = math.ceil(args.sample_size / args.num_jobs)

    output_dir = Path(args.output_dir) / args.job_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for job_id in range(args.num_jobs):
        start_idx = job_id * sentences_per_job
        end_idx = min((job_id + 1) * sentences_per_job, args.sample_size)

        script = generate_job_script(
            job_id + 1,
            start_idx,
            end_idx,
            args.num_jobs,
            args.exp_name,
        )

        script_path = output_dir / f"job_{job_id + 1}_of_{args.num_jobs}.sh"
        with open(script_path, "w") as f:
            f.write(script)

        script_path.chmod(0o755)

    print(f"Generated {args.num_jobs} job scripts in {output_dir}")
    print(f"Each job will process approximately {sentences_per_job} sentences")
    print("You can submit all jobs with:")
    print(f"for f in {output_dir}/*.sh; do sbatch $f; done")


if __name__ == "__main__":
    main()
