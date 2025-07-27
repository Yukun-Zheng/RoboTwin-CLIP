# CLIP+PCA集成到DP策略的使用说明

本文档说明如何在RoboTwin的DP策略中使用CLIP图像编码器和PCA降维功能。

## 功能特性

- **CLIP图像编码器**: 使用预训练的CLIP模型替换原有的ResNet18进行图像特征提取
- **PCA降维**: 将CLIP特征降维到指定维度（默认8维）
- **参数冻结**: CLIP编码器参数在训练过程中保持冻结
- **PCA一致性**: 训练和评估使用相同的PCA坐标系
- **可配置参数**: CLIP模型类型和PCA维度均可配置

## 新增文件

1. `diffusion_policy/model/vision/clip_pca_encoder.py` - CLIP+PCA编码器实现
2. `diffusion_policy/config/robot_dp_clip_14.yaml` - CLIP配置文件
3. `train_clip.sh` - CLIP训练脚本
4. `eval_clip.sh` - CLIP评估脚本

## 支持的CLIP模型

- `ViT-B/32` (默认)
- `ViT-B/16`
- `ViT-L/14`
- `RN50`
- `RN101`
- `RN50x4`
- `RN50x16`
- `RN50x64`

## 使用方法

### 1. 环境准备

```bash
# 激活环境
conda activate yukun-RoboTwin

# 确保安装了CLIP库
pip install git+https://github.com/openai/CLIP.git
```

### 2. 训练策略

使用新的训练脚本 `train_clip.sh`：

```bash
# 基本用法（使用默认CLIP模型ViT-B/32和8维PCA）
bash train_clip.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}

# 指定CLIP模型和PCA维度
bash train_clip.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id} ${clip_model_name} ${pca_dim}

# 示例：使用ViT-B/32模型，PCA降维到8维
bash train_clip.sh place_shoe demo_clean 50 0 14 6 "ViT-B/32" 8

# 示例：使用ViT-B/16模型，PCA降维到16维
bash train_clip.sh place_shoe demo_clean 50 0 14 6 "ViT-B/16" 16

# 示例：使用默认参数
bash train_clip.sh beat_block_hammer demo_randomized 50 0 14 0
```

### 3. 评估策略

使用新的评估脚本 `eval_clip.sh`：

```bash
# 基本用法
bash eval_clip.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}

# 指定CLIP模型和PCA维度（必须与训练时一致）
bash eval_clip.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id} ${clip_model_name} ${pca_dim}

# 示例：评估使用ViT-B/16和16维PCA训练的模型
bash eval_clip.sh beat_block_hammer demo_clean demo_randomized 50 0 0 "ViT-B/16" 16

# 示例：使用默认参数评估
bash eval_clip.sh beat_block_hammer demo_clean demo_randomized 50 0 0
```

## 参数说明

### 训练脚本参数

- `task_name`: 任务名称
- `task_config`: 任务配置
- `expert_data_num`: 专家数据数量
- `seed`: 随机种子
- `action_dim`: 动作维度（如aloha-agilex为14）
- `gpu_id`: GPU设备ID（0-7）
- `clip_model_name`: CLIP模型名称（可选，默认"ViT-B/32"）
- `pca_dim`: PCA降维维度（可选，默认8）

### 评估脚本参数

- `task_name`: 任务名称
- `task_config`: 评估环境配置
- `ckpt_setting`: 训练数据配置
- `expert_data_num`: 专家数据数量
- `seed`: 随机种子
- `gpu_id`: GPU设备ID（0-7）
- `clip_model_name`: CLIP模型名称（必须与训练时一致）
- `pca_dim`: PCA降维维度（必须与训练时一致）

## 技术细节

### CLIP特征提取

- 输入图像自动从HDF5格式转换为CLIP所需的RGB格式
- 使用CLIP内置的图像预处理（resize、center crop、归一化）
- CLIP编码器参数在训练过程中保持冻结

### PCA降维

- 在训练开始时，收集一定数量的CLIP特征用于PCA拟合
- PCA模型保存到检查点目录，评估时自动加载
- 训练和评估使用相同的PCA坐标系确保一致性

### 内存优化

- 批处理大小从128减少到64以适应CLIP的内存需求
- 支持梯度累积以保持有效批处理大小

## 注意事项

1. **参数一致性**: 评估时必须使用与训练时相同的CLIP模型和PCA维度
2. **GPU内存**: CLIP模型可能需要更多GPU内存，建议使用较小的批处理大小
3. **PCA拟合**: 首次训练时需要收集足够的特征进行PCA拟合，可能会稍微延长训练开始时间
4. **模型下载**: 首次使用CLIP模型时会自动下载，需要网络连接

## 故障排除

### 常见问题

1. **CLIP模型下载失败**: 检查网络连接，或手动下载模型文件
2. **GPU内存不足**: 减少批处理大小或使用更小的CLIP模型
3. **PCA维度错误**: 确保评估时使用与训练时相同的PCA维度
4. **配置文件未找到**: 确保`robot_dp_clip_14.yaml`文件存在

### 调试模式

在训练脚本中设置`DEBUG=True`可以启用调试模式，使用离线wandb记录。

## 性能对比

相比原始ResNet18编码器：
- **优势**: 更强的视觉表示能力，预训练知识迁移
- **劣势**: 更高的计算和内存开销
- **适用场景**: 需要更好视觉理解的复杂任务