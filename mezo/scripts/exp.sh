device=$1
export CUDA_VISIBLE_DEVICES=$device

if [ $device -eq 0 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=SST2 bash scripts/mezo.sh --early_stop
    done

elif [ $device -eq 1 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=RTE bash scripts/mezo.sh --early_stop
    done

elif [ $device -eq 2 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=CB bash scripts/mezo.sh --early_stop
    done

elif [ $device -eq 3 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=BoolQ bash scripts/mezo.sh --early_stop
    done

elif [ $device -eq 4 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=Copa bash scripts/mezo.sh --early_stop
    done

elif [ $device -eq 5 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=MultiRC bash scripts/mezo.sh --early_stop
    done

elif [ $device -eq 6 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=SQuAD bash scripts/mezo.sh --early_stop
    done

elif [ $device -eq 7 ];then
    for lr in 1e-07 1e-06;do
        LR=$lr TASK=DROP bash scripts/mezo.sh --early_stop
    done
fi