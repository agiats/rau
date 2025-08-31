. experiments/non_anonymous_config.bash
SINGULARITY_IMAGE_FILE=/cluster/work/cotterell/tsomeya/container-images/lm_inductive_bias.sif
DATA_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/data
RESULTS_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/results
RAU_DIR=/cluster/home/tsomeya/projects/lm_inductive_bias/src/rau

FIGURES_DIR=$RESULTS_DIR/figures
PERTURBATION_CONFIG_FILE=config/perturbation_func.json
ARCHITECTURES=("lstm" "transformer" "stack-rnn_lstm_64_superposition-20" "stack-transformer_20-1.superposition-16.1")
ARCHITECTURE_LABELS=("LSTM" "Transformer" "RNN+Sup. 20" "Tf+Sup. 16")

NUM_TRIALS=5

# PFSA dataset generation parameters
N_SYMBOLS_LIST=(32 48 64)
N_STATES_LIST=(16 24 32)
N_TOPOLOGY_SEEDS=5
N_WEIGHT_SEEDS=5

submit_job() {
  bash experiments/submit_job.bash "$@"
}
