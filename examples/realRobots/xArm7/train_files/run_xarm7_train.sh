#!/usr/bin/env bash
# Train StarVLA QwenOFT on the xArm7 data. Run from the repository root in the starvla env:
#     bash examples/realRobots/xArm7/train_files/run_xarm7_train.sh
# Every training setting is in the yaml below; this script only launches it and passes nothing that overrides it.
set -euo pipefail

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

config_yaml=./examples/realRobots/xArm7/train_files/starvla_qwenoft_xarm7.yaml
num_gpus=2      # the one thing that is not in the yaml: accelerate needs it before the config is read
export RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)   # appended to run_id by the yaml, so every launch gets its own output directory

# the wandb account the TARFlow runs use (the project is wandb_project in the yaml)
# export WANDB_MODE=disabled


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_gpus} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml}
