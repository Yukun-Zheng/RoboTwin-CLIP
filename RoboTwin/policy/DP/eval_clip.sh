#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
clip_model_name=${7:-"ViT-B/32"}  # 默认使用ViT-B/32
pca_dim=${8:-8}  # 默认PCA降维到8维
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mCLIP model: ${clip_model_name}\033[0m"
echo -e "\033[33mPCA dimension: ${pca_dim}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --clip_model_name "${clip_model_name}" \
    --pca_dim ${pca_dim}