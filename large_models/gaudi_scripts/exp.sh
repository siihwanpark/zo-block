device=$1

if [ $device -eq 0 ]; then
    for task in RTE BoolQ WIC Copa MultiRC ReCoRD SQuAD DROP;do
        if [ $task == "BoolQ" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ] || [ $task == "SQuAD" ] || [ $task == "DROP" ];then
            steps=1000
        else
            steps=2000
        fi

        for lr in 1e-07 2e-07;do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps $steps --momentum 0.0 --save_perturbations
        done
    done

elif [ $device -eq 1 ]; then
    for task in RTE BoolQ WSC WIC Copa;do
        if [ $task == "BoolQ" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ] || [ $task == "SQuAD" ] || [ $task == "DROP" ];then
            steps=1000
        else
            steps=2000
        fi

        for lr in 1e-06 2e-06;do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps $steps --momentum 0.0 --save_perturbations\
                --sparse_perturbation --sparse_perturbation_type random --gradient_sparsity 0.75
        done
    done

elif [ $device -eq 2 ]; then
    for task in MultiRC ReCoRD SQuAD DROP;do
        if [ $task == "BoolQ" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ] || [ $task == "SQuAD" ] || [ $task == "DROP" ];then
            steps=1000
        else
            steps=2000
        fi

        for lr in 1e-06 2e-06;do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps $steps --momentum 0.0 --save_perturbations\
                --sparse_perturbation --sparse_perturbation_type random --gradient_sparsity 0.75
        done
    done

elif [ $device -eq 3 ]; then
    for task in CB RTE BoolQ WSC WIC Copa;do
        if [ $task == "BoolQ" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ] || [ $task == "SQuAD" ] || [ $task == "DROP" ];then
            steps=1000
        else
            steps=2000
        fi

        for lr in 5e-07 1e-06;do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps $steps --momentum 0.0 --save_perturbations\
                --sparse_perturbation --sparse_perturbation_type scale --gradient_sparsity 0.75
        done
    done
elif [ $device -eq 4 ]; then
    for task in MultiRC ReCoRD SQuAD DROP;do
        if [ $task == "BoolQ" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ] || [ $task == "SQuAD" ] || [ $task == "DROP" ];then
            steps=1000
        else
            steps=2000
        fi

        for lr in 5e-07 1e-06;do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps $steps --momentum 0.0 --save_perturbations\
                --sparse_perturbation --sparse_perturbation_type scale --gradient_sparsity 0.75
        done
    done

elif [ $device -eq 5 ]; then
    for task in SST2 RTE CB BoolQ WSC WIC Copa MultiRC ReCoRD SQuAD DROP;do
        if [ $task == "BoolQ" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ] || [ $task == "SQuAD" ] || [ $task == "DROP" ];then
            steps=1000
        else
            steps=2000
        fi

        for lr in 1e-07 2e-07;do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps $steps --momentum 0.0 --save_perturbations\
                --lozo_perturbation --rank_r 2 --lowrank_step_interval 100
        done
    done
elif [ $device -eq 6 ]; then
    for task in RTE CB BoolQ WSC WIC Copa MultiRC ReCoRD SQuAD DROP;do
        if [ $task == "BoolQ" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ] || [ $task == "SQuAD" ] || [ $task == "DROP" ];then
            steps=1000
        else
            steps=2000
        fi

        for lr in 2e-06 5e-06 1e-05;do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps $steps --momentum 0.0 --save_perturbations\
                --bcd --bcd_order random --bcd_interval 100
        done
    done
fi
