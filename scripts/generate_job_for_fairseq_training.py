import argparse
from pathlib import Path


def generate_preprocess_script(
    grammar_name: str,
    exp_name: str,
) -> str:
    return f"""#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name={exp_name}_preprocess_{grammar_name}
#SBATCH --output=logs/{exp_name}/preprocess/{grammar_name}.out
#SBATCH --error=logs/{exp_name}/preprocess/{grammar_name}.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \\
    --only-source \\
    --trainpref "data/fairseq_train/{exp_name}/{grammar_name}/train.txt" \\
    --validpref "data/fairseq_train/{exp_name}/{grammar_name}/dev.txt" \\
    --testpref "data/fairseq_train/{exp_name}/{grammar_name}/test.txt" \\
    --destdir "data/fairseq_train_preprocessed/{exp_name}/{grammar_name}" \\
    --workers 4
"""


def generate_job_script(
    model_type: str,
    exp_name: str,
    grammar_name: str,
    gpu_id: int = 0,
    seed: int = 0,
) -> str:
    # モデル固有の設定
    if model_type == "transformer":
        arch = "transformer_lm_gpt"
        results_dir = "transformer_results"
        checkpoints_dir = "transformer_checkpoints"
    elif model_type == "transformer_tiny":
        arch = "transformer_lm_gpt2_tiny"
        results_dir = "transformer_tiny_results"
        checkpoints_dir = "transformer_tiny_checkpoints"
    elif model_type == "transformer_4layer":
        arch = "transformer_lm_gpt"
        results_dir = "transformer_4layer_results"
        checkpoints_dir = "transformer_4layer_checkpoints"
    else:  # lstm
        arch = "lstm_lm"
        results_dir = "lstm_results"
        checkpoints_dir = "lstm_checkpoints"

    return f"""#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16g
#SBATCH --gres=gpumem:20g
#SBATCH --tmp=20g
#SBATCH --job-name={exp_name}_lm_{model_type}_{grammar_name}_seed_{seed}
#SBATCH --output=logs/{exp_name}/lm_{model_type}/{grammar_name}_seed_{seed}.out
#SBATCH --error=logs/{exp_name}/lm_{model_type}/{grammar_name}_seed_{seed}.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="{gpu_id}"

# Train model
fairseq-train \\
    --task language_modeling \\
    "data/fairseq_train_preprocessed/{exp_name}/{grammar_name}" \\
    --save-dir "results/{checkpoints_dir}/{exp_name}/{grammar_name}/seed{seed}" \\
    --arch {arch} \\
    --share-decoder-input-output-embed \\
    --dropout 0.3 \\
    --optimizer adam \\
    --adam-betas '(0.9,0.98)' \\
    --weight-decay 0.01 \\
    --lr 0.0005 \\
    --lr-scheduler inverse_sqrt \\
    --warmup-init-lr 1e-07 \\
    --tokens-per-sample 512 \\
    --sample-break-mode none \\
    --max-tokens 2048 \\
    --patience 10 \\
    --no-epoch-checkpoints \\
    --no-last-checkpoints \\
    --seed {seed} {"--decoder-layers 4" if model_type == "transformer_4layer" else ""}

mkdir -p "results/{results_dir}/{exp_name}/{grammar_name}/seed{seed}"
# Evaluate on dev set
fairseq-eval-lm \\
    "data/fairseq_train_preprocessed/{exp_name}/{grammar_name}" \\
    --path "results/{checkpoints_dir}/{exp_name}/{grammar_name}/seed{seed}/checkpoint_best.pt" \\
    --tokens-per-sample 512 \\
    --gen-subset "valid" \\
    --output-word-probs \\
    --quiet 2> "results/{results_dir}/{exp_name}/{grammar_name}/seed{seed}/dev.txt"

# Evaluate on test set
fairseq-eval-lm \\
    "data/fairseq_train_preprocessed/{exp_name}/{grammar_name}" \\
    --path "results/{checkpoints_dir}/{exp_name}/{grammar_name}/seed{seed}/checkpoint_best.pt" \\
    --tokens-per-sample 512 \\
    --gen-subset "test" \\
    --output-word-probs \\
    --quiet 2> "results/{results_dir}/{exp_name}/{grammar_name}/seed{seed}/test.txt"

# Calculate sentence scores
python scripts/get_sentence_scores.py \\
    -i "results/{results_dir}/{exp_name}/{grammar_name}/seed{seed}/test.txt" \\
    -O "results/{results_dir}/{exp_name}/{grammar_name}/seed{seed}/test.scores.txt"
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
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name to generate jobs for",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        help="Model type to generate jobs for (default: transformer)",
    )
    parser.add_argument(
        "--grammar_name",
        type=str,
        required=True,
        help="Grammar name to process",
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
    preprocess_dir = base_output_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    script = generate_preprocess_script(
        grammar_name=args.grammar_name,
        exp_name=args.exp_name,
    )
    script_path = preprocess_dir / f"{args.grammar_name}.sh"
    with open(script_path, "w") as f:
        f.write(script)
    script_path.chmod(0o755)

    # Generate training scripts
    output_dir = base_output_dir / f"lm_{args.model_type}" / args.grammar_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.num_seeds):
        script = generate_job_script(
            model_type=args.model_type,
            exp_name=args.exp_name,
            grammar_name=args.grammar_name,
            gpu_id=0,
            seed=seed,
        )

        script_path = output_dir / f"train_{seed}.sh"
        with open(script_path, "w") as f:
            f.write(script)
        script_path.chmod(0o755)

    print(f"Generated job scripts in {base_output_dir}")


if __name__ == "__main__":
    main()
