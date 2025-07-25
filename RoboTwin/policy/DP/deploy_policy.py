import numpy as np
from .dp_model import DP
import yaml

# 部署DP模型的配置文件
# 将环境的观测转换为模型可以接受的格式
# 还可以重置模型的观测

# 将观测转换为模型可以接受的格式
def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs

# 获取模型，可以是DP模型，也可以是其他模型
def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    # Check if using CLIP model
    if 'clip_model' in usr_args and usr_args['clip_model']:
        load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}_clip.yaml'
        print(f"\033[32mUsing CLIP model: {usr_args['clip_model']}\033[0m")
    else:
        load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
        print(f"\033[32mUsing standard DP model\033[0m")
    
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    # Pass clip_model parameter if provided
    clip_model = usr_args.get('clip_model', None)
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, clip_model=clip_model)

# 功能是获取模型动作，然后执行动作，最后返回动作。
def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== 获取动作 ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

# 重置模型，使其观测归零
def reset_model(model):
    model.reset_obs()
