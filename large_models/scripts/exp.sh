device=$1

# if [ $device -eq 0 ]; then
#     # Sparse random perturbation experiments
#     for group in layer global;do
#         for p in 0.2 0.25;do
#             for interval in 1 10;do
#                 for lr in 1e-05 2e-05 5e-05;do
#                     bash scripts/run.sh $device MeZO-Adam\
#                         --sparse_perturbation --sparse_gradient_group $group\
#                         --gradient_sparsity $p --sparse_gradient_resample_steps $interval\
#                         --learning_rate $lr
#                 done
#             done
#         done
#     done

# elif [ $device -eq 1 ]; then
#     for rank_r in 1 2 4;do
#         for nu in 50 100;do
#             for lr in 1e-05 2e-05 5e-05;do
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --lozo_perturbation --rank_r $rank_r --lowrank_step_interval $nu\
#                     --learning_rate $lr
#             done
#         done
#     done

# elif [ $device -eq 2 ]; then
#     for rank_r in 24 48;do
#         for nu in 500 1000;do
#             for lr in 1e-05 2e-05 5e-05;do
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --subzero_perturbation --rank_r $rank_r --lowrank_step_interval $nu\
#                     --learning_rate $lr
                
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --subzero_perturbation --rank_r $rank_r --lowrank_step_interval $nu\
#                     --learning_rate $lr --orthonormal_projection
#             done
#         done
#     done

# elif [ $device -eq 3 ]; then
#     for rank_r in 1 2 4 16 32 64;do
#         for nu in 50 100 500 1000;do
#             for lr in 1e-05 2e-05 5e-05;do
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --kfac_perturbation --rank_r $rank_r --lowrank_step_interval $nu\
#                     --learning_rate $lr
                
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --kfac_perturbation --rank_r $rank_r --lowrank_step_interval $nu\
#                     --learning_rate $lr --orthonormal_projection
#             done
#         done
#     done

# elif [ $device -eq 4 ]; then
#     for k in 50 100 200 500;do
#         for order in random ascending;do
#             for lr in 1e-05 2e-05 5e-05 1e-04;do
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --badam --badam_K $k --badam_ordering $order\
#                     --learning_rate $lr --max_steps 40000
#             done
#         done
#     done

# elif [ $device -eq 5 ]; then
#     for lr in 1e-05 2e-05 5e-05 1e-04;do
#         bash scripts/run.sh $device MeZO-Adam\
#             --badam --badam_K 500 --badam_ordering random\
#             --learning_rate $lr --max_steps 40000\
#             --sparse_perturbation --sparse_gradient_group layer --gradient_sparsity 0.2 --sparse_gradient_resample_steps 1
        
#         bash scripts/run.sh $device MeZO-Adam\
#             --badam --badam_K 500 --badam_ordering random\
#             --learning_rate $lr --max_steps 40000\
#             --sparse_perturbation --sparse_gradient_group global --gradient_sparsity 0.2 --sparse_gradient_resample_steps 1

#         bash scripts/run.sh $device MeZO-Adam\
#             --badam --badam_K 500 --badam_ordering random\
#             --learning_rate $lr --max_steps 40000\
#             --lozo_perturbation --rank_r 1 --lowrank_step_interval 50
        
#         bash scripts/run.sh $device MeZO-Adam\
#             --badam --badam_K 500 --badam_ordering random\
#             --learning_rate $lr --max_steps 40000\
#             --subzero_perturbation --rank_r 24 --lowrank_step_interval 1000

#         bash scripts/run.sh $device MeZO-Adam\
#             --badam --badam_K 500 --badam_ordering random\
#             --learning_rate $lr --max_steps 40000\
#             --subzero_perturbation --rank_r 24 --lowrank_step_interval 1000 --orthonormal_projection
#     done

# elif [ $device -eq 6 ]; then
#     for lr in 1e-05 2e-05 5e-05 1e-04;do
#         for rank_r in 1 2 4 16 32 64;do
#             for nu in 50 100 500 1000;do
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --badam --badam_K 500 --badam_ordering random\
#                     --learning_rate $lr --max_steps 40000\
#                     --kfac_perturbation --rank_r $rank_r --lowrank_step_interval $nu
#             done
#         done
#     done

# elif [ $device -eq 7 ]; then
#     for lr in 1e-05 2e-05 5e-05 1e-04;do
#         for rank_r in 1 2 4 16 32 64;do
#             for nu in 50 100 500 1000;do
#                 bash scripts/run.sh $device MeZO-Adam\
#                     --badam --badam_K 500 --badam_ordering random\
#                     --learning_rate $lr --max_steps 40000\
#                     --kfac_perturbation --rank_r $rank_r --lowrank_step_interval $nu --orthonormal_projection
#             done
#         done
#     done
# fi

if [ $device -eq 0 ]; then
    for lr in 5e-07 1e-06;do
        for K in 10 50 100;do
            for order in random ascending;do
                bash scripts/run.sh $device MeZO-SGD\
                    --badam --badam_K $K --badam_ordering $order\
                    --learning_rate $lr --max_steps 30000 --include_embedding --include_lm_head
            done
        done
    done

elif [ $device -eq 1 ]; then
    for lr in 1e-06 2e-06 5e-06 1e-05;do
        bash scripts/run.sh $device MeZO-Lion\
            --learning_rate $lr --max_steps 20000 --beta1 0.9 --beta2 0.99 --cautious_optimizer

        # bash scripts/run.sh $device MeZO-Lion\
        #     --learning_rate $lr --max_steps 20000 --beta1 0.95 --beta2 0.98 --cautious_optimizer
    done
elif [ $device -eq 2 ]; then
    for lr in 5e-06 1e-05 5e-05 1e-04;do
        K=50
        order=ascending
        bash scripts/run.sh $device MeZO-Lion\
            --badam --badam_K $K --badam_ordering $order\
            --learning_rate $lr --max_steps 30000 --beta1 0.9 --beta2 0.99 --cautious_optimizer

        # bash scripts/run.sh $device MeZO-Lion\
        #     --badam --badam_K $K --badam_ordering $order\
        #     --learning_rate $lr --max_steps 30000 --beta1 0.95 --beta2 0.98 --cautious_optimizer
    done
elif [ $device -eq 3 ]; then
    for lr in 5e-06 1e-05 5e-05 1e-04;do
        for K in 10 50 100;do
            for order in random ascending;do
                bash scripts/run.sh $device MeZO-SGD\
                    --badam --badam_K $K --badam_ordering $order\
                    --learning_rate $lr --max_steps 30000 --include_embedding --include_lm_head --momentum 0.0
            done
        done
    done
elif [ $device -eq 4 ]; then
    for lr in 1e-06 5e-06 1e-05 5e-05 1e-04;do
        bash scripts/run.sh $device MeZO-Lion\
            --badam --badam_K $K --badam_ordering $order\
            --learning_rate $lr --max_steps 30000 --beta1 0.9 --beta2 0.99

        bash scripts/run.sh $device MeZO-Lion\
            --badam --badam_K $K --badam_ordering $order\
            --learning_rate $lr --max_steps 30000 --beta1 0.95 --beta2 0.98
    done
elif [ $device -eq 5 ]; then
    for lr in 1e-06 5e-06 1e-05 5e-05 1e-04;do
        bash scripts/run.sh $device MeZO-Lion\
            --learning_rate $lr --max_steps 20000 --beta1 0.9 --beta2 0.99

        bash scripts/run.sh $device MeZO-Lion\
            --learning_rate $lr --max_steps 20000 --beta1 0.95 --beta2 0.98
    done
elif [ $device -eq 6 ]; then
    for lr in 5e-05 7e-05 1e-04 2e-04;do
        for K in 5 10 20;do
            for order in random ascending;do
                bash scripts/run.sh $device MeZO-Adam\
                    --badam --badam_K $K --badam_ordering $order\
                    --learning_rate $lr --state_flush --max_steps 50000 --include_embedding --include_lm_head --fine_blocks
            done
        done
    done
elif [ $device -eq 7 ]; then
    for lr in 5e-05 7e-05 1e-04 2e-04;do
        for K in 50 75 100;do
            for order in random ascending;do
                bash scripts/run.sh $device MeZO-Adam\
                    --badam --badam_K $K --badam_ordering $order\
                    --learning_rate $lr --state_flush --max_steps 50000 --include_embedding --include_lm_head --fine_blocks
            done
        done
    done
fi