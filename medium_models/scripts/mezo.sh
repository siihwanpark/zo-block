#!/bin/bash
# -*- coding: utf-8 -*-
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export TQDM_DISABLE=1

TASK=${TASK:-SNLI}
K=${K:-16}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-100000}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-"../../../Model/roberta-large"}
MODELNAME=${MODELNAME:-"roberta-large"}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "Step: $STEP; Eval step: $EVAL_STEP"

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=${TAG:-k${K}-${MODELNAME}-mezo-${EXTRA_TAG}}

echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

mkdir -p log_dir
# Redirect all output to a log file based on the TAG
exec &> >(tee "log_dir/${TASK}-${GR_TAG}-${TAG}.log")

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash scripts/run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP --weight_decay $WD --zero_order_eps $EPS \
    --zero_order_optim --lr_scheduler_type "constant" --optimizer "sgd" --efficient_zero_order \
    $@
