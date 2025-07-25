# CLIP集成到DP策略 - 最小化修改方案

## 概述

本方案通过最小化修改现有DP代码，实现CLIP模型的集成。用户可以使用几乎相同的命令来训练和评估CLIP增强的DP模型。

## 安装依赖

首先安装CLIP：
```bash
pip install git+https://github.com/openai/CLIP.git
```

## 使用方法

### 1. 训练模型

**标准DP模型（原有方式）：**
```bash
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}
# 例如：
bash train.sh put_bottles_dustbin demo_randomized 50 0 14 0
```

**CLIP增强DP模型（新增一个参数）：**
```bash
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id} true
# 例如：
bash train.sh put_bottles_dustbin demo_randomized 50 0 14 0 true
```

### 2. 评估模型

**标准DP模型：**
```bash
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
# 例如：
bash eval.sh put_bottles_dustbin demo_randomized demo_randomized 50 0 0
```

**CLIP增强DP模型：**
```bash
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id} true
# 例如：
bash eval.sh put_bottles_dustbin demo_randomized demo_randomized 50 0 0 true
```

## 修改内容

### 1. 核心文件修改

- **`model_getter.py`**: 添加了`get_clip()`函数，支持加载CLIP视觉编码器
- **`multi_image_obs_encoder.py`**: 添加了CLIP模型自动检测和预处理逻辑
- **`train.sh`**: 添加了可选的第7个参数来选择是否使用CLIP
- **`eval.sh`**: 添加了可选的第7个参数来选择是否使用CLIP

### 2. 新增配置文件

- **`robot_dp_14_clip.yaml`**: 14维动作空间的CLIP配置
- **`robot_dp_16_clip.yaml`**: 16维动作空间的CLIP配置

## CLIP配置选项

在配置文件中，可以调整以下CLIP相关参数：

```yaml
rgb_model:
  _target_: diffusion_policy.model.vision.model_getter.get_clip
  name: "ViT-L/14@336px"  # CLIP模型名称："ViT-B/32", "ViT-L/14@336px", "RN50"等
  freeze: True            # 是否冻结CLIP参数
  feature_layer: -1       # 提取特征的层数（-1为最终层，-2为倒数第二层）
```

## 技术细节

### 1. 自动预处理
- 当检测到CLIP模型时，自动应用正确的图像尺寸调整（ViT-L/14@336px使用336x336）
- 自动启用ImageNet归一化
- 禁用随机裁剪（对CLIP更适合）

### 2. 训练优化
- 降低batch size（64 vs 128）以适应CLIP的内存需求
- 降低学习率（5e-5 vs 1e-4）以适应预训练模型
- 增加warmup步数（1000 vs 500）以稳定训练

### 3. 特征维度
- ViT-B/32: 512维特征
- ViT-L/14@336px: 768维特征（当前使用）
- RN50: 1024维特征

## 预期改进

1. **更强的视觉理解能力**：CLIP的预训练提供了丰富的视觉-语言知识
2. **更好的泛化性能**：预训练模型在新场景下表现更稳定
3. **更高的任务成功率**：特别是在复杂视觉场景中

## 调试建议

1. **内存不足**：降低batch size或使用较小的CLIP模型（如ViT-B/32）
2. **训练不稳定**：增加warmup步数或降低学习率
3. **性能下降**：尝试不同的feature_layer设置或调整freeze参数

## 实验对比

建议进行以下对比实验：
1. 标准DP vs CLIP-DP在相同数据集上的性能
2. 不同CLIP模型（ViT-B/32, ViT-L/14）的效果对比
3. 冻结vs微调CLIP参数的影响
4. 不同特征层提取的效果对比

## 注意事项

1. 确保CLIP库正确安装
2. CLIP模型首次使用时会自动下载，需要网络连接
3. CLIP模型比ResNet18更大，训练时间会增加
4. 评估时确保使用与训练时相同的CLIP配置