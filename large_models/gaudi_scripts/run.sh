#!/bin/bash

export PT_HPU_LAZY_MODE=0
export PT_HPU_GPU_MIGRATION=1

trainer=$1
task=$2
shift 2

TASK=$task
Trainer=$trainer

MODEL=${MODEL:-facebook/opt-13b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
LR=${LR:-1e-7}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-500}
MODE=${MODE:-ft}

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
echo "Trainer:         $Trainer"
echo "Steps:           $STEPS"
echo "Eval steps:      $EVAL_STEPS"
echo "Mode:            $MODE"
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
echo "Num Sampling:    $NUM_SAMPLING"
echo "================================================="

# Execute script
python run_gaudi.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL}-$TAG --tag $TAG \
    --train_set_seed $SEED \
    --num_train $TRAIN --num_dev $DEV --num_eval $EVAL \
    --logging_steps 1 \
    --max_steps $STEPS \
    --trainer $Trainer --load_float16 \
    --learning_rate $LR --zo_eps $EPS \
    --per_device_train_batch_size $BS \
    --lr_scheduler_type "constant" \
    --evaluation_strategy steps --save_strategy no \
    --eval_steps $EVAL_STEPS \
    --train_as_classification \
    --max_grad_norm 0.0 \
    --early_stop\
    $EXTRA_ARGS \
    $TASK_ARGS \
    $@