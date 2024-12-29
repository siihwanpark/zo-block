device=$1

# bash scripts/run.sh $device MeZO-Adam\
#     --sparse_perturbation --sparse_gradient_group layer --gradient_sparsity 0.2 --sparse_gradient_resample_steps 1

# bash scripts/run.sh $device MeZO-Adam\
#     --sparse_perturbation --sparse_gradient_group global --gradient_sparsity 0.2 --sparse_gradient_resample_steps 1

# bash scripts/run.sh $device MeZO-Adam\
#     --subzero_perturbation --rank_r 24 --lowrank_step_interval 1000

# bash scripts/run.sh $device MeZO-Adam\
#     --lozo_perturbation --rank_r 1 --lowrank_step_interval 50

# bash scripts/run.sh $device MeZO-Adam\
#     --kfac_perturbation --rank_r 24 --lowrank_step_interval 1000 --orthonormal_projection

# bash scripts/run.sh $device MeZO-Adam\
#     --badam --badam_ordering random --badam_K 100

bash scripts/run.sh $device MeZO-Adam\
    --rht_perturbation --reverse_rht --rht_step_interval 1