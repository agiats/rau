#!/bin/bash
set -euo pipefail

. experiments/include.bash

# Config
OUTPUT_ROOT=${OUTPUT_ROOT:-"$DATA_DIR/LM_random_sampling"}
ARCHS=(${ARCHS:-lstm transformer})
VOCAB_SIZES=(${VOCAB_SIZES:-32 48 64})
SEEDS=(${SEEDS:-1 2 3 4 5})

# Sampling sizes
NUM_TRAIN=${NUM_TRAIN:-20000}
NUM_VAL=${NUM_VAL:-5000}
NUM_TEST=${NUM_TEST:-5000}
MAX_LENGTH=${MAX_LENGTH:-128}

# Resource requests
TIME=${TIME:-12:00:00}
MEM_PER_CPU=${MEM_PER_CPU:-8g}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

mkdir -p "$OUTPUT_ROOT"
mkdir -p logs

for arch in "${ARCHS[@]}"; do
  for vocab in "${VOCAB_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      job_name="lm_sample_${arch}_V${vocab}_S${seed}"
      case "$arch" in
        lstm)
          cmd=(
            bash experiments/language_sampling/sample_lstm_random.bash \
              "$OUTPUT_ROOT" "$vocab" \
              "$NUM_TRAIN" "$NUM_VAL" "$NUM_TEST" \
              "$MAX_LENGTH" "$seed" "$seed"
          )
          ;;
        transformer)
          cmd=(
            bash experiments/language_sampling/sample_transformer_random.bash \
              "$OUTPUT_ROOT" "$vocab" \
              "$NUM_TRAIN" "$NUM_VAL" "$NUM_TEST" \
              "$MAX_LENGTH" "$seed" "$seed"
          )
          ;;
        *)
          echo "Unknown arch: $arch" >&2; exit 1;
          ;;
      esac
    #   submit_job \
    #     "$job_name" \
    #     cpu \
    #     --time="$TIME" \
    #     --mem-per-cpu="$MEM_PER_CPU" \
    #     --cpus-per-task="$CPUS_PER_TASK" \
    #     -- \
    poetry run "${cmd[@]}"
    done
  done
done

echo "Submitted jobs to output root: $OUTPUT_ROOT"


