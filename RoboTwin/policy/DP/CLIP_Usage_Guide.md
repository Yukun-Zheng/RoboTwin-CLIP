# CLIP模型参数化使用指南

本指南说明如何在训练和评估时动态选择不同的CLIP模型。

## 支持的CLIP模型

根据OpenAI CLIP的官方模型列表，支持以下模型：

- `RN50` - ResNet-50基础模型
- `RN101` - ResNet-101模型
- `RN50x4` - ResNet-50 4倍宽度
- `RN50x16` - ResNet-50 16倍宽度
- `RN50x64` - ResNet-50 64倍宽度
- `ViT-B/32` - Vision Transformer Base，32x32 patch
- `ViT-B/16` - Vision Transformer Base，16x16 patch
- `ViT-L/14` - Vision Transformer Large，14x14 patch
- `ViT-L/14@336px` - Vision Transformer Large，336px输入分辨率（默认）

## 使用方法

### 训练模型

#### 标准DP模型（无CLIP）
```bash
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}
```

#### CLIP增强模型（使用默认ViT-L/14@336px）
```bash
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id} true
```

#### CLIP增强模型（指定特定模型）
```bash
# 使用ViT-B/32模型
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id} true "ViT-B/32"

# 使用ResNet-50模型
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id} true "RN50"

# 使用ViT-L/14模型
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id} true "ViT-L/14"
```

### 评估模型

#### 标准DP模型（无CLIP）
```bash
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}
```

#### CLIP增强模型（使用默认ViT-L/14@336px）
```bash
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id} true
```

#### CLIP增强模型（指定特定模型）
```bash
# 使用ViT-B/32模型
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id} true "ViT-B/32"

# 使用ResNet-50模型
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id} true "RN50"
```

## 参数说明

- 第7个参数：`use_clip` - 是否使用CLIP模型（`true`/`false`，默认`false`）
- 第8个参数：`clip_model` - CLIP模型名称（可选，默认`"ViT-L/14@336px"`）

## 技术细节

### 自动预处理
- 系统会根据CLIP模型名称自动调整图像预处理
- 包含"336px"的模型使用336x336输入尺寸
- 其他模型使用224x224输入尺寸
- 自动应用ImageNet归一化

### 模型特征维度
- ResNet模型：1024维（RN50）
- ViT-B模型：512维
- ViT-L模型：768维

### 配置文件
- 使用CLIP时自动选择`robot_dp_{action_dim}_clip.yaml`配置
- 不使用CLIP时使用`robot_dp_{action_dim}.yaml`配置

## 注意事项

1. **模型名称必须准确**：请确保使用的模型名称与OpenAI CLIP支持的模型完全一致
2. **内存需求**：较大的模型（如ViT-L）需要更多GPU内存
3. **训练时间**：不同模型的训练时间可能有所差异
4. **兼容性**：评估时使用的CLIP模型应与训练时使用的模型一致

## 示例

```bash
# 训练一个使用ViT-B/16的CLIP增强模型
bash train.sh place_object_basket default 100 42 14 0 true "ViT-B/16"

# 评估对应的模型
bash eval.sh place_object_basket default latest 100 42 0 true "ViT-B/16"
```

这种设计允许您轻松比较不同CLIP模型的性能，同时保持代码的简洁性。