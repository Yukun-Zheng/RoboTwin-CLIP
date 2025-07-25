#!/bin/bash

# 具体流程是
# 1. 从config中读取参数
# 2. 从config中读取policy的参数
# 3. 从config中读取task的参数
# 4. 从config中读取ckpt的参数
# 5. 从config中读取expert_data的参数
# 6. 从config中读取seed的参数
# 7. 从config中读取gpu的参数
# 8. 从config中读取debug的参数
# 9. 设置CUDA_VISIBLE_DEVICES
# 10. 输出gpu id
# 11. 切换到上级目录
# 12. 调用eval_policy.py

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
use_clip=${7:-false}  # Optional 7th parameter, defaults to false
clip_model=${8:-"ViT-L/14@336px"}  # Optional 8th parameter, defaults to ViT-L/14@336px
DEBUG=False

# Check if using CLIP model
if [ "$use_clip" = "true" ] || [ "$use_clip" = "True" ] || [ "$use_clip" = "1" ]; then
    echo -e "\033[32mEvaluating CLIP-enhanced DP model\033[0m"
else
    echo -e "\033[32mEvaluating standard DP model\033[0m"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    $([ "$use_clip" = "true" ] || [ "$use_clip" = "True" ] || [ "$use_clip" = "1" ] && echo "--clip_model ${clip_model}" || echo "")