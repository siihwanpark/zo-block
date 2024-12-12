#!/bin/bash

export PT_HPU_LAZY_MODE=0
export PT_HPU_GPU_MIGRATION=1

# Default values
MODEL="facebook/opt-1.3b"
BS=16
EPS=1e-3
TRAIN=1000
DEV=500
EVAL=1000
STEPS=20000
EVAL_STEPS=100
MODE="ft"
TASK="SST2"
SEED=0
RANK=1
STEP_INTERVAL=100
Trainer="KFAC-LOZO-Adam"
LR=1e-4
SAVE_STEPS=1000

# Additional arguments based on mode
case $MODE in
    prefix)
        EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
        ;;
    lora)
        EXTRA_ARGS="--lora"
        ;;
    *)
        EXTRA_ARGS=""
        ;;
esac

# Task-specific arguments
case $TASK in
    CB|Copa)
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD|DROP|SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    *)
        TASK_ARGS=""
        ;;
esac

# Generate tag
TAG="$Trainer-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$STEP_INTERVAL-$RANK"

# Print configuration
echo "================= Configuration ================="
echo "Model:           $MODEL"
echo "Task:            $TASK"
echo "Batch size:      $BS"
echo "Learning rate:   $LR"
echo "EPS:             $EPS"
echo "Seed:            $SEED"
echo "Train/Eval steps: $STEPS/$EVAL_STEPS"
echo "Mode:            $MODE"
echo "Extra args:      $EXTRA_ARGS $TASK_ARGS"
echo "Rank:            $RANK"
echo "Step interval:   $STEP_INTERVAL"
echo "Tag:             $TAG"
echo "================================================="

# Execute script
python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL}-$TAG --tag $TAG \
    --train_set_seed $SEED \
    --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --logging_steps 1 \
    --max_steps $STEPS \
    --trainer $Trainer --load_bfloat16 \
    --learning_rate $LR --zo_eps $EPS \
    --per_device_train_batch_size $BS \
    --lr_scheduler_type "constant" \
    --load_best_model_at_end \
    --evaluation_strategy steps --save_strategy steps \
    --save_total_limit 1 \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --train_as_classification \
    --step_interval $STEP_INTERVAL \
    --rank_r $RANK \
    --use_hpu\
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
