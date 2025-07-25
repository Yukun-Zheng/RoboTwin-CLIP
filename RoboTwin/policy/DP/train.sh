#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
use_clip=${7:-false}  # Optional 7th parameter, defaults to false
clip_model=${8:-"ViT-L/14@336px"}  # Optional 8th parameter, defaults to ViT-L/14@336px

head_camera_type=D435

DEBUG=False
save_ckpt=True

# Choose config based on whether to use CLIP
if [ "$use_clip" = "true" ] || [ "$use_clip" = "True" ] || [ "$use_clip" = "1" ]; then
    alg_name=robot_dp_${action_dim}_clip
    config_name=robot_dp_${action_dim}_clip
    addition_info=train_clip
    echo -e "\033[32mUsing CLIP-enhanced DP model\033[0m"
else
    alg_name=robot_dp_$action_dim
    config_name=${alg_name}
    addition_info=train
    echo -e "\033[32mUsing standard DP model\033[0m"
fi
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type \
                            $([ "$use_clip" = "true" ] || [ "$use_clip" = "True" ] || [ "$use_clip" = "1" ] && echo "policy.obs_encoder.rgb_model.name=${clip_model}" || echo "")
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \