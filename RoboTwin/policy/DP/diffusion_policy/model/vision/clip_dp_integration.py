import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union
import copy
import sys
import os

# 添加CLIP和RoboTwin路径
sys.path.append('/home/lumina/lumina/yukun/CLIP')
sys.path.append('/home/lumina/lumina/yukun/RoboTwin')

import clip
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply

class CLIPImageEncoder(nn.Module):
    """
    CLIP图像编码器，用于替换DP中的ResNet编码器
    支持提取CLIP的中间层特征或最终特征
    """
    def __init__(self, 
                 model_name="ViT-B/32", 
                 layer=11, 
                 use_final_features=False,
                 feature_dim=512,
                 freeze_clip=True):
        super().__init__()
        
        # 加载CLIP模型
        self.clip_model, _ = clip.load(model_name, device='cpu')
        self.layer = layer
        self.use_final_features = use_final_features
        self.feature_dim = feature_dim
        
        # 是否冻结CLIP参数
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # 如果使用中间层特征，需要添加适配层
        if not use_final_features:
            # ViT-B/32的中间层特征维度是768
            self.feature_adapter = nn.Sequential(
                nn.Linear(768, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            # CLIP最终特征维度是512
            if feature_dim != 512:
                self.feature_adapter = nn.Linear(512, feature_dim)
            else:
                self.feature_adapter = nn.Identity()
    
    def extract_intermediate_features(self, x):
        """
        提取CLIP的中间层特征
        """
        visual = self.clip_model.visual
        
        # 确保数据类型匹配
        x = x.to(visual.conv1.weight.dtype)
        
        # Patch embedding
        x = visual.conv1(x)  # shape: [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape: [*, grid ** 2, width]
        
        # 添加class token
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 通过指定层数的transformer blocks
        for i in range(self.layer + 1):
            x = visual.transformer.resblocks[i](x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 使用class token作为图像特征
        cls_token = x[:, 0, :]  # [batch_size, 768]
        
        return cls_token
    
    def forward(self, x):
        """
        前向传播
        x: [batch_size, 3, H, W] 输入图像
        返回: [batch_size, feature_dim] 特征向量
        """
        if self.use_final_features:
            # 使用CLIP的最终特征
            features = self.clip_model.encode_image(x)
        else:
            # 使用中间层特征
            features = self.extract_intermediate_features(x)
        
        # 通过适配层
        features = self.feature_adapter(features)
        
        return features

class CLIPMultiImageObsEncoder(ModuleAttrMixin):
    """
    集成CLIP的多图像观测编码器
    替换DP中的MultiImageObsEncoder
    """
    def __init__(self,
                 shape_meta: dict,
                 clip_model_name: str = "ViT-B/32",
                 clip_layer: int = 11,
                 use_final_features: bool = False,
                 feature_dim: int = 512,
                 freeze_clip: bool = True,
                 share_rgb_model: bool = False,
                 resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = (224, 224),
                 imagenet_norm: bool = True):
        """
        Args:
            shape_meta: 形状元数据
            clip_model_name: CLIP模型名称
            clip_layer: 使用的CLIP层数（如果use_final_features=False）
            use_final_features: 是否使用CLIP的最终特征
            feature_dim: 输出特征维度
            freeze_clip: 是否冻结CLIP参数
            share_rgb_model: 是否共享RGB模型
            resize_shape: 图像resize尺寸
            imagenet_norm: 是否使用ImageNet归一化
        """
        super().__init__()
        
        # 创建CLIP编码器
        self.clip_encoder = CLIPImageEncoder(
            model_name=clip_model_name,
            layer=clip_layer,
            use_final_features=use_final_features,
            feature_dim=feature_dim,
            freeze_clip=freeze_clip
        )
        
        # 解析观测形状
        rgb_keys = []
        low_dim_keys = []
        key_shape_map = {}
        
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map
        self.share_rgb_model = share_rgb_model
        self.feature_dim = feature_dim
        
        # 图像预处理
        if resize_shape is not None:
            self.resize = nn.AdaptiveAvgPool2d(resize_shape)
        else:
            self.resize = nn.Identity()
        
        # ImageNet归一化
        if imagenet_norm:
            self.normalize = lambda x: F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = nn.Identity()
    
    def forward(self, obs_dict):
        """
        前向传播
        obs_dict: 观测字典，包含rgb图像和低维状态
        """
        batch_size = None
        features = []
        
        # 处理RGB输入
        for key in self.rgb_keys:
            img = obs_dict[key]  # [B, 3, H, W]
            
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            
            # 图像预处理
            img = self.resize(img)
            img = self.normalize(img)
            
            # 通过CLIP编码器
            feature = self.clip_encoder(img)  # [B, feature_dim]
            features.append(feature)
        
        # 处理低维输入
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            features.append(data)
        
        # 拼接所有特征
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        """
        计算输出形状
        """
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros((batch_size,) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

def create_clip_dp_config():
    """
    创建集成CLIP的DP配置
    """
    config = {
        'policy': {
            'obs_encoder': {
                '_target_': 'clip_dp_integration.CLIPMultiImageObsEncoder',
                'clip_model_name': 'ViT-B/32',
                'clip_layer': 11,
                'use_final_features': False,  # 使用中间层特征
                'feature_dim': 512,
                'freeze_clip': True,
                'share_rgb_model': False,
                'resize_shape': (224, 224),  # CLIP标准输入尺寸
                'imagenet_norm': True
            }
        }
    }
    return config

if __name__ == "__main__":
    # 测试CLIP编码器
    print("测试CLIP图像编码器...")
    
    # 创建测试数据
    batch_size = 2
    test_img = torch.randn(batch_size, 3, 224, 224)
    
    # 测试CLIP编码器
    clip_encoder = CLIPImageEncoder(
        model_name="ViT-B/32",
        layer=11,
        use_final_features=False,
        feature_dim=512
    )
    
    with torch.no_grad():
        features = clip_encoder(test_img)
        print(f"CLIP编码器输出形状: {features.shape}")
    
    # 测试多图像观测编码器
    print("\n测试CLIP多图像观测编码器...")
    
    shape_meta = {
        "obs": {
            "head_cam": {"shape": [3, 224, 224], "type": "rgb"},
            "left_cam": {"shape": [3, 224, 224], "type": "rgb"},
            "right_cam": {"shape": [3, 224, 224], "type": "rgb"},
            "agent_pos": {"shape": [14], "type": "low_dim"}
        }
    }
    
    clip_obs_encoder = CLIPMultiImageObsEncoder(
        shape_meta=shape_meta,
        clip_model_name="ViT-B/32",
        feature_dim=512
    )
    
    # 创建测试观测
    test_obs = {
        "head_cam": torch.randn(batch_size, 3, 224, 224),
        "left_cam": torch.randn(batch_size, 3, 224, 224), 
        "right_cam": torch.randn(batch_size, 3, 224, 224),
        "agent_pos": torch.randn(batch_size, 14)
    }
    
    with torch.no_grad():
        encoded_obs = clip_obs_encoder(test_obs)
        print(f"编码后观测形状: {encoded_obs.shape}")
        print(f"特征维度: RGB特征 {3 * 512} + 低维特征 {14} = {3 * 512 + 14}")
    
    print("\n集成测试完成！")