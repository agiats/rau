Language sampling (RAU)

Overview
- Generate random samples from untrained random LMs (LSTM/Transformer) using RAU.
- Produces PFSA-like directories: main.vocab, main.tok, datasets/{validation,test}/main.tok, metadata.json, model/{parameters.pt, kwargs.json}.

Sample size
- Per split, the number of sequences is specified by arguments:
  - num_train, num_val, num_test
- These map to RAU generate's --num-samples for a single call per split.
- Sequence length is bounded by max_length (tokens, including EOS when sampled).

Scripts
- sample_lstm_random.bash
  - Creates dummy vocab (tokens "0..V-1", allow_unk=False), initializes a random LSTM, then samples.
  - Usage:
    - env: RAU_DIR must point to the RAU source tree (e.g., "$PWD/src/rau").
    - args: <output_root_dir> <vocab_size> <num_train> <num_val> <num_test> <max_length> <parameter_seed> <random_seed>
    - example:
      export RAU_DIR="$PWD/src/rau"
      out="$PWD/work/lm_local_test/lstm_V8"
      poetry run bash experiments/language_sampling/sample_lstm_random.bash "$out" 8 100 20 20 128 123 1

- sample_transformer_random.bash
  - Same as LSTM but initializes a transformer (small default dims for local tests).
  - Usage is identical to the LSTM script.

- sample_lm_random.bash (internal helper)
  - Called by the per-arch scripts.
  - Args: <training_data_dir> <saved_model_dir> <num_train> <num_val> <num_test> <max_length> <random_seed>
  - Performs one RAU generation call per split with --num-samples set to the requested count.

- submit_sampling_jobs.bash (HPC)
  - Submits jobs over architectures, vocab sizes, and seeds.
  - Config via env:
    - OUTPUT_ROOT (default: "$DATA_DIR/LM/random_sampling")
    - ARCHS (default: lstm transformer)
    - VOCAB_SIZES (e.g., 32 48 64)
    - SEEDS (e.g., 1 2 3 4 5)
    - NUM_TRAIN, NUM_VAL, NUM_TEST, MAX_LENGTH
    - TIME, MEM_PER_CPU, CPUS_PER_TASK
  - Example:
    export ARCHS=("lstm" "transformer")
    export VOCAB_SIZES=(32 48)
    export SEEDS=(1 2)
    export OUTPUT_ROOT="$DATA_DIR/LM/random_sampling"
    export NUM_TRAIN=20000 NUM_VAL=5000 NUM_TEST=5000 MAX_LENGTH=128
    bash experiments/language_sampling/submit_sampling_jobs.bash

Outputs
- <output_root>/<arch_tag>/
  - main.vocab: torch-saved dict with {tokens: ["0".."V-1"], allow_unk: false}
  - main.tok: num_train lines
  - datasets/validation/main.tok: num_val lines
  - datasets/test/main.tok: num_test lines
  - metadata.json: provenance and generation parameters
  - model/{kwargs.json, parameters.pt}: saved random model

Notes
- We do not run prepare_data here; separate scripts exist if you need .prepared files.
- EOS is automatically added internally by RAU; do not include it in tokens.
- Randomness: parameter_seed controls model initialization; random_seed controls sampling RNG.

Troubleshooting
- If python cannot import RAU, ensure RAU_DIR is set to the local RAU source (e.g., export RAU_DIR="$PWD/src/rau").
- If kwargs.json is missing for older/newer RAU versions, the initializer writes it alongside parameters.pt.


