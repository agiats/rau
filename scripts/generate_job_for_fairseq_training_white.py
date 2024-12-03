import argparse
from pathlib import Path


def generate_job_script(
    model_type: str,
    grammar_name: str,
    split_id: int,
    gpu_id: int = 0,
) -> str:
    # モデル固有の設定
    if model_type == "transformer":
        arch = "transformer_lm_gpt2_small"
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
#SBATCH --job-name=eos_zipf_min1_max_20_lm_{model_type}_{grammar_name}_split_{split_id}
#SBATCH --output=logs/eos_zipf_min1_max_20/lm_{model_type}/{grammar_name}/split_{split_id}.out
#SBATCH --error=logs/eos_zipf_min1_max_20/lm_{model_type}/{grammar_name}/split_{split_id}.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="{gpu_id}"

# Preprocess data
fairseq-preprocess \\
    --only-source \\
    --trainpref "data/fairseq_train/eos_zipf_min1_max_20/{grammar_name}/split_{split_id}.train" \\
    --validpref "data/fairseq_train/eos_zipf_min1_max_20/{grammar_name}/split_{split_id}.dev" \\
    --testpref "data/fairseq_train/eos_zipf_min1_max_20/{grammar_name}/split_{split_id}.test" \\
    --destdir "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/{grammar_name}/split_{split_id}" \\
    --workers 10

    # Train model
fairseq-train \\
    --task language_modeling \\
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/{grammar_name}/split_{split_id}" \\
    --save-dir "results/{checkpoints_dir}/{grammar_name}/split_{split_id}" \\
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
    --patience 5 \\
    --max-update 10000 \\
    --no-epoch-checkpoints \\
    --no-last-checkpoints

mkdir -p "results/{results_dir}/{grammar_name}"
# Evaluate on dev set
fairseq-eval-lm \\
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/{grammar_name}/split_{split_id}" \\
    --path "results/{checkpoints_dir}/{grammar_name}/split_{split_id}/checkpoint_best.pt" \\
    --tokens-per-sample 512 \\
    --gen-subset "valid" \\
    --output-word-probs \\
    --quiet 2> "results/{results_dir}/{grammar_name}/split_{split_id}.dev.txt"

# Evaluate on test set
fairseq-eval-lm \\
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/{grammar_name}/split_{split_id}" \\
    --path "results/{checkpoints_dir}/{grammar_name}/split_{split_id}/checkpoint_best.pt" \\
    --tokens-per-sample 512 \\
    --gen-subset "test" \\
    --output-word-probs \\
    --quiet 2> "results/{results_dir}/{grammar_name}/split_{split_id}.test.txt"

# Calculate sentence scores
python scripts/get_sentence_scores.py \\
    -i "results/{results_dir}/{grammar_name}/split_{split_id}.test.txt" \\
    -O "results/{results_dir}/{grammar_name}/split_{split_id}.test.scores.txt"
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
        "--num_splits",
        type=int,
        default=10,
        help="Number of splits per grammar (default: 10)",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["transformer", "lstm"],
        help="Model types to generate jobs for (default: transformer lstm)",
    )
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)

    # 各モデルタイプとgrammarの組み合わせでジョブを生成
    for model_type in args.model_types:
        for grammar_name in args.grammar_names:
            output_dir = base_output_dir / f"lm_{model_type}" / grammar_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for split_id in range(args.num_splits):
                script = generate_job_script(
                    model_type=model_type,
                    grammar_name=grammar_name,
                    split_id=split_id,
                    gpu_id=0,  # GPUはスケジューラに任せる
                )

                script_path = output_dir / f"job_split_{split_id}.sh"
                with open(script_path, "w") as f:
                    f.write(script)

                script_path.chmod(0o755)

    print(f"Generated job scripts in {base_output_dir}")
    print(f"for f in $(find {base_output_dir} -name '*.sh'); do sbatch $f; done")


if __name__ == "__main__":
    main()
