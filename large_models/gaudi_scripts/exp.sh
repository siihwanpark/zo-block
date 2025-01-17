device=$1

# if [ $device -eq 0 ]; then
#     for task in SST2 CB WSC SQuAD DROP;do
#         bash gaudi_scripts/run.sh MeZO-SGD $task\
#             --learning_rate 1e-07 --max_steps 20000 --momentum 0.0 --save_perturbations
#     done

# elif [ $device -eq 1 ]; then
#     for task in RTE BoolQ WIC Copa MultiRC ReCoRD;do
#         bash gaudi_scripts/run.sh MeZO-SGD $task\
#             --learning_rate 1e-06 --max_steps 20000 --momentum 0.0 --save_perturbations
#     done

# elif [ $device -eq 2 ]; then
#     for task in SST2 CB WSC SQuAD DROP;do
#         bash gaudi_scripts/run.sh MeZO-SGD $task\
#             --learning_rate 1e-07 --max_steps 20000 --momentum 0.0 --save_perturbations\
#             --lozo_perturbation --rank_r 2 --lowrank_step_interval 100
#     done

# elif [ $device -eq 3 ]; then
#     for task in RTE BoolQ WIC Copa MultiRC ReCoRD;do
#         bash gaudi_scripts/run.sh MeZO-SGD $task\
#             --learning_rate 1e-06 --max_steps 20000 --momentum 0.0 --save_perturbations\
#             --lozo_perturbation --rank_r 2 --lowrank_step_interval 100
#     done

# elif [ $device -eq 4 ]; then
#     for task in SST2 CB WSC SQuAD DROP;do
#         bash gaudi_scripts/run.sh MeZO-SGD $task\
#             --learning_rate 5e-06 --max_steps 20000 --momentum 0.0 --save_perturbations\
#             --bcd --bcd_ordering random --bcd_interval 100
#     done

# elif [ $device -eq 5 ]; then
#     for task in RTE BoolQ WIC Copa MultiRC ReCoRD;do
#         bash gaudi_scripts/run.sh MeZO-SGD $task\
#             --learning_rate 5e-05 --max_steps 20000 --momentum 0.0 --save_perturbations\
#             --bcd --bcd_ordering random --bcd_interval 100
#     done

if [ $device -eq 0 ]; then
    # for task in SST2 MultiRC;do
    #     for lr in 1e-07 1e-06;do
    #         bash gaudi_scripts/run.sh MeZO-SGD $task\
    #             --learning_rate $lr --max_steps 10000 --momentum 0.0 --save_perturbations
    #     done
    # done

    bash gaudi_scripts/run.sh MeZO-SGD SST2\
        --learning_rate 1e-07 --max_steps 10000 --momentum --save_perturbations\
        --seed 0

elif [ $device -eq 1 ]; then
    # for task in RTE SQuAD;do
    #     for lr in 1e-07 1e-06;do
    #         bash gaudi_scripts/run.sh MeZO-SGD $task\
    #             --learning_rate $lr --max_steps 10000 --momentum 0.0 --save_perturbations
    #     done
    # done

    bash gaudi_scripts/run.sh MeZO-SGD SST2\
        --learning_rate 1e-07 --max_steps 10000 --momentum --save_perturbations\
        --seed 1

elif [ $device -eq 2 ]; then
    # for task in CB WIC;do
    #     for lr in 1e-07 1e-06;do
    #         bash gaudi_scripts/run.sh MeZO-SGD $task\
    #             --learning_rate $lr --max_steps 10000 --momentum 0.0 --save_perturbations
    #     done
    # done

    bash gaudi_scripts/run.sh MeZO-SGD SST2\
        --learning_rate 1e-07 --max_steps 10000 --momentum --save_perturbations\
        --seed 2

elif [ $device -eq 3 ]; then
    # for task in BoolQ Copa;do
    #     for lr in 1e-07 1e-06;do
    #         bash gaudi_scripts/run.sh MeZO-SGD $task\
    #             --learning_rate $lr --max_steps 10000 --momentum 0.0 --save_perturbations
    #     done
    # done

    bash gaudi_scripts/run.sh MeZO-SGD SST2\
        --learning_rate 1e-07 --max_steps 10000 --momentum --save_perturbations\
        --seed 3

elif [ $device -eq 4 ]; then
    # for task in WSC ReCoRD;do
    #     for lr in 1e-07 1e-06;do
    #         bash gaudi_scripts/run.sh MeZO-SGD $task\
    #             --learning_rate $lr --max_steps 10000 --momentum 0.0 --save_perturbations
    #     done
    # done

    bash gaudi_scripts/run.sh MeZO-SGD SST2\
        --learning_rate 1e-07 --max_steps 10000 --momentum --save_perturbations\
        --seed 4

elif [ $device -eq 5 ]; then
    # for task in DROP;do
    #     for lr in 1e-07 1e-06;do
    #         bash gaudi_scripts/run.sh MeZO-SGD $task\
    #             --learning_rate $lr --max_steps 10000 --momentum 0.0 --save_perturbations
    #     done
    # done

    bash gaudi_scripts/run.sh MeZO-SGD SST2\
        --learning_rate 1e-07 --max_steps 10000 --momentum --save_perturbations\
        --seed 5

elif [ $device -eq 6 ]; then
    for task in SST2 RTE CB BoolQ WSC WIC Copa MultiRC ReCoRD SQuAD;do
        if [ $task == "RTE" ] || [ $task == "BoolQ" ] || [ $task == "WIC" ] || [ $task == "Copa" ] || [ $task == "MultiRC" ] || [ $task == "ReCoRD" ];then
            lrs=(3e-07 1e-06 3e-06)
        else
            lrs=(3e-07 1e-06 3e-06)
        fi

        for lr in ${lrs[@]};do
            bash gaudi_scripts/run.sh MeZO-SGD $task\
                --learning_rate $lr --max_steps 1000 --momentum 0.0 --save_perturbations\
                --sparse_perturbation --sparse_perturbation_type scale --gradient_sparsity 0.75
        done
    done

    for lr in 1e-07 1e-06;do
        bash gaudi_scripts/run.sh MeZO-SGD DROP\
            --learning_rate $lr --max_steps 1000 --momentum 0.0\
            --sparse_perturbation --sparse_perturbation_type scale --gradient_sparsity 0.75
    done
fi
