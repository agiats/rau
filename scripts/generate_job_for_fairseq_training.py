import argparse
from pathlib import Path


def generate_preprocess_script(
    grammar_name: str,
) -> str:
    return f"""#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=eos_zipf_min1_max_20_100K_dedup_preprocess_{grammar_name}
#SBATCH --output=logs/eos_zipf_min1_max_20_100K_dedup/preprocess/{grammar_name}.out
#SBATCH --error=logs/eos_zipf_min1_max_20_100K_dedup/preprocess/{grammar_name}.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \\
    --only-source \\
    --trainpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/{grammar_name}/train.txt" \\
    --validpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/{grammar_name}/dev.txt" \\
    --testpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/{grammar_name}/test.txt" \\
    --destdir "data/fairseq_train_preprocessed/eos_zipf_min1_max_20_100K_dedup/{grammar_name}" \\
    --workers 64
"""


def generate_job_script(
    model_type: str,
    grammar_name: str,
    gpu_id: int = 0,
    seed: int = 0,
) -> str:
    # モデル固有の設定
    if model_type == "transformer":
        arch = "transformer_lm_gpt"
        results_dir = "transformer_results"
        checkpoints_dir = "transformer_checkpoints"
    else:  # lstm
        arch = "lstm_lm"
        results_dir = "lstm_results"
        checkpoints_dir = "lstm_checkpoints"

    return f"""#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=eos_zipf_min1_max_20_100K_dedup_lm_{model_type}_{grammar_name}_seed_{seed}
#SBATCH --output=logs/eos_zipf_min1_max_20_100K_dedup/lm_{model_type}/{grammar_name}_seed_{seed}.out
#SBATCH --error=logs/eos_zipf_min1_max_20_100K_dedup/lm_{model_type}/{grammar_name}_seed_{seed}.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="{gpu_id}"

# Train model
fairseq-train \\
    --task language_modeling \\
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20_100K_dedup/{grammar_name}" \\
    --save-dir "results/{checkpoints_dir}/{grammar_name}_100K_dedup/seed{seed}" \\
    --arch {arch} \\
    --share-decoder-input-output-embed \\
    --dropout 0.3 \\
    --optimizer adam \\
    --adam-betas '(0.9,0.98)' \\
    --weight-decay 0.01 \\
    --lr 0.0005 \\
    --lr-scheduler inverse_sqrt \\
    --warmup-updates 4000 \\
    --clip-norm 0.0 \\
    --warmup-init-lr 1e-07 \\
    --tokens-per-sample 512 \\
    --sample-break-mode none \\
    --max-tokens 2048 \\
    --update-freq 16 \\
    --patience 3 \\
    --max-update 10000 \\
    --no-epoch-checkpoints \\
    --no-last-checkpoints \\
    --seed {seed}

mkdir -p "results/{results_dir}/{grammar_name}_100K_dedup/seed{seed}"
# Evaluate on dev set
fairseq-eval-lm \\
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20_100K_dedup/{grammar_name}" \\
    --path "results/{checkpoints_dir}/{grammar_name}_100K_dedup/seed{seed}/checkpoint_best.pt" \\
    --tokens-per-sample 512 \\
    --gen-subset "valid" \\
    --output-word-probs \\
    --quiet 2> "results/{results_dir}/{grammar_name}_100K_dedup/seed{seed}/dev.txt"

# Evaluate on test set
fairseq-eval-lm \\
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20_100K_dedup/{grammar_name}" \\
    --path "results/{checkpoints_dir}/{grammar_name}_100K_dedup/seed{seed}/checkpoint_best.pt" \\
    --tokens-per-sample 512 \\
    --gen-subset "test" \\
    --output-word-probs \\
    --quiet 2> "results/{results_dir}/{grammar_name}_100K_dedup/seed{seed}/test.txt"

# Calculate sentence scores
python scripts/get_sentence_scores.py \\
    -i "results/{results_dir}/{grammar_name}_100K_dedup/seed{seed}/test.txt" \\
    -O "results/{results_dir}/{grammar_name}_100K_dedup/seed{seed}/test.scores.txt"
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate language model training job scripts"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scripts/jobs",
        help="Directory to store the generated job scripts",
    )
    parser.add_argument(
        "--grammar_names",
        nargs="+",
        required=True,
        help="List of grammar names to process",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["transformer", "lstm"],
        help="Model types to generate jobs for (default: transformer lstm)",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of seeds to use for each model training",
    )
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    # Generate preprocess scripts
    preprocess_dir = base_output_dir / "preprocess_100K_dedup"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    for grammar_name in args.grammar_names:
        script = generate_preprocess_script(grammar_name=grammar_name)
        script_path = preprocess_dir / f"{grammar_name}.sh"
        with open(script_path, "w") as f:
            f.write(script)
        script_path.chmod(0o755)

    # Generate training scripts
    for model_type in args.model_types:
        for grammar_name in args.grammar_names:
            output_dir = base_output_dir / f"lm_{model_type}_100K_dedup" / grammar_name
            output_dir.mkdir(parents=True, exist_ok=True)
            for seed in range(args.num_seeds):
                script = generate_job_script(
                    model_type=model_type,
                    grammar_name=grammar_name,
                    gpu_id=0,
                    seed=seed,
                )

                script_path = output_dir / f"train_{seed}.sh"
                with open(script_path, "w") as f:
                    f.write(script)
                script_path.chmod(0o755)

    print(f"Generated job scripts in {base_output_dir}")
    print("To submit preprocess jobs:")
    print(
        f"for f in $(find {base_output_dir}/preprocess -name '*.sh'); do sbatch $f; done"
    )
    print("\nTo submit training jobs:")
    print(
        f"for f in $(find {base_output_dir} -path '*/lm_*' -name '*.sh'); do sbatch $f; done"
    )


if __name__ == "__main__":
    main()
