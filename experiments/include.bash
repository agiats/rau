SINGULARITY_IMAGE_FILE=/cluster/work/cotterell/bdusell/container-images/private-nn-formal-languages.sif
DATA_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/data
RESULTS_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/results
RAU_DIR=/cluster/home/tsomeya/projects/lm_inductive_bias/src/rau
FIGURES_DIR=$RESULTS_DIR/figures

EXP_NAMES=("local_entropy_disjoint" "local_entropy_non_disjoint")
ARCHITECTURES=("lstm" "transformer")

NUM_SEEDS=5
submit_job() {
  bash experiments/submit_job.bash "$@"
}
