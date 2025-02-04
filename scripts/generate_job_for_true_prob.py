import json
import math
import argparse
from pathlib import Path


def generate_job_script(
    grammar_file: str,
    job_id: int,
    start_idx: int,
    end_idx: int,
    total_jobs: int,
    log_dir: str,
    sentence_counts_path: str,
    output_dir: str,
) -> str:
    return f"""#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --output=logs/{log_dir}/split_{job_id}_of_{total_jobs}.out
#SBATCH --error=logs/{log_dir}/split_{job_id}_of_{total_jobs}.err

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

n_processes=64

if [ ! -f "{output_dir}/lower_bound_entropy.value" ]; then
    python scripts/calculate_true_prob.py \\
        --grammar_file {grammar_file} \\
        --start_symbol "S" \\
        --normalize \\
        --sentence_counts_path {sentence_counts_path} \\
        --output_path {output_dir}/probability_split_{job_id}_of_{total_jobs}.csv.gz \\
        --num_workers $n_processes \\
        --start_index {start_idx} \\
        --end_index {end_idx}
fi
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate split job scripts for sentence processing"
    )
    parser.add_argument(
        "--grammar_file",
        type=str,
        required=True,
        help="Path to the grammar file",
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
        "--log_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sentence_counts_path",
        type=str,
        required=True,
        help="Path to the sentence counts file",
    )
    parser.add_argument(
        "--job_output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--sample_size", type=int, required=True, help="Number of sentences to sample"
    )
    args = parser.parse_args()

    sentences_per_job = math.ceil(args.sample_size / args.num_jobs)

    output_dir = Path(args.job_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for job_id in range(args.num_jobs):
        start_idx = job_id * sentences_per_job
        end_idx = min((job_id + 1) * sentences_per_job, args.sample_size)

        script = generate_job_script(
            args.grammar_file,
            job_id + 1,
            start_idx,
            end_idx,
            args.num_jobs,
            args.log_dir,
            args.sentence_counts_path,
            args.output_dir,
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
