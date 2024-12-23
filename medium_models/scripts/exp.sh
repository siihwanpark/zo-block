# !/bin/bash

device=$1
export CUDA_VISIBLE_DEVICES=$device

if [ $device -eq 0 ]; then
    # TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large RANK=4 STEP_INTERVAL=100 bash scripts/lozo.sh
    TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-5 EPS=1e-3 MODEL=roberta-large bash scripts/mezo.sh --v_t_logging_steps 10 --optimizer adam --zero_order_use_trainer_optim
elif [ $device -eq 1 ]; then
    TASK=SST-2 K=16 SEED=42 BS=64 LR=1e-4 EPS=1e-3 MODEL=roberta-large bash scripts/mezo.sh
fi