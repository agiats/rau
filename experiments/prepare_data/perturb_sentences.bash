set -euo pipefail
. experiments/include.bash

exp_name="deterministic_shuffles"
data_name="BLLIP_MD_longer_than_10"
EXP_DIR="$DATA_DIR"/"$data_name"/"$exp_name"
BASE_DIR="$EXP_DIR"/Base

SPLIT_CONFIG_DIR="$RESULTS_DIR"/"$data_name"/"$exp_name"/split_configs
mkdir -p "$SPLIT_CONFIG_DIR"

N_SPLITS=70
python src/perturbation/split_config.py \
    "$PERTURBATION_CONFIG_FILE" \
    "$SPLIT_CONFIG_DIR" \
    "$N_SPLITS"


for config_file in "$SPLIT_CONFIG_DIR"/config_*.json; do
    job_suffix=$(basename "$config_file" .json)
    submit_job \
        "perturb_sentences+${data_name}+${exp_name}+${job_suffix}" \
        cpu \
        --tasks=48 \
        --mem-per-cpu=8g \
        --time=8:00:00 \
        -- \
        python perturbation/perturb_sentences.py \
        --base_train_file "$BASE_DIR"/train.txt \
        --base_dev_file "$BASE_DIR"/dev.txt \
        --base_test_file "$BASE_DIR"/test.txt \
        --exp_dir "$EXP_DIR" \
        --perturb_config_file "$config_file" \
        --n_workers 48
done



