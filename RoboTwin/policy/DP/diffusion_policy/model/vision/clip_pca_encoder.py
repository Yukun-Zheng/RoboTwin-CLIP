import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
from PIL import Image
import sys

# 添加CLIP路径
clip_path = "/data0/lumina/yukun/CLIP"
if clip_path not in sys.path:
    sys.path.append(clip_path)

import clip

class CLIPPCAEncoder(nn.Module):
    """
    CLIP图像编码器 + PCA降维模块
    
    该模块使用CLIP的image encoder提取图像特征，然后通过PCA降维到指定维度。
    CLIP的参数被冻结，只有PCA组件在训练时会更新。
    
    Args:
        clip_model_name (str): CLIP模型名称，如'ViT-B/32', 'RN50'等
        pca_dim (int): PCA降维后的维度，默认为8
        device (str): 设备类型，默认为'cuda'
        pca_save_path (str): PCA模型保存路径
    """
    
    def __init__(self, clip_model_name='ViT-B/32', pca_dim=8, device='cuda', pca_save_path=None):
        super().__init__()
        
        self.clip_model_name = clip_model_name
        self.pca_dim = pca_dim
        self.device = device
        self.pca_save_path = pca_save_path or f"./pca_models/pca_{clip_model_name.replace('/', '_')}_{pca_dim}d.pkl"
        
        # 加载CLIP模型
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        
        # 冻结CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 获取CLIP图像编码器的输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_features = self.clip_model.encode_image(dummy_input)
            self.clip_output_dim = dummy_features.shape[-1]
        
        print(f"CLIP output dimension: {self.clip_output_dim}")
        print(f"Target PCA dimension: {pca_dim}")
        
        # PCA组件
        self.pca = None
        self.pca_fitted = False
        
        # 用于存储训练时的特征，以便拟合PCA
        self.training_features = []
        
    def _preprocess_images(self, images):
        """
        预处理图像以适配CLIP输入格式
        
        Args:
            images (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        # 如果输入图像不是224x224，需要调整尺寸
        if images.shape[-2:] != (224, 224):
            images = torch.nn.functional.interpolate(
                images, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # CLIP期望的归一化参数
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device).view(1, 3, 1, 1)
        
        # 归一化到[0,1]然后应用CLIP归一化
        if images.max() > 1.0:
            images = images / 255.0
        
        images = (images - mean) / std
        
        return images
    
    def _extract_clip_features(self, images):
        """
        使用CLIP提取图像特征
        
        Args:
            images (torch.Tensor): 预处理后的图像张量
            
        Returns:
            torch.Tensor: CLIP特征
        """
        with torch.no_grad():
            # 确保输入图像与CLIP模型在同一设备和数据类型
            images = images.to(self.device).type(self.clip_model.dtype)
            features = self.clip_model.encode_image(images)
            # 归一化特征
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def fit_pca(self, dataloader=None, max_samples=10000):
        """
        使用训练数据拟合PCA
        
        Args:
            dataloader: 数据加载器，如果为None则使用存储的训练特征
            max_samples (int): 用于拟合PCA的最大样本数
        """
        print("Fitting PCA...")
        
        if dataloader is not None:
            # 从数据加载器收集特征
            all_features = []
            sample_count = 0
            
            self.eval()
            with torch.no_grad():
                for batch in dataloader:
                    if sample_count >= max_samples:
                        break
                    
                    # 假设batch包含图像数据
                    if isinstance(batch, dict):
                        # 处理多相机输入
                        for key in ['head_cam', 'left_cam', 'right_cam']:
                            if key in batch:
                                images = batch[key]
                                if len(images.shape) == 5:  # (B, T, C, H, W)
                                    B, T = images.shape[:2]
                                    images = images.view(-1, *images.shape[2:])
                                
                                images = self._preprocess_images(images)
                                features = self._extract_clip_features(images)
                                all_features.append(features.cpu().numpy())
                                sample_count += features.shape[0]
                    else:
                        images = batch[0] if isinstance(batch, (list, tuple)) else batch
                        if len(images.shape) == 5:  # (B, T, C, H, W)
                            B, T = images.shape[:2]
                            images = images.view(-1, *images.shape[2:])
                        
                        images = self._preprocess_images(images)
                        features = self._extract_clip_features(images)
                        all_features.append(features.cpu().numpy())
                        sample_count += features.shape[0]
            
            if all_features:
                all_features = np.concatenate(all_features, axis=0)
            else:
                raise ValueError("No features collected from dataloader")
        
        elif self.training_features:
            # 使用存储的训练特征
            all_features = np.concatenate(self.training_features, axis=0)
        else:
            raise ValueError("No training features available for PCA fitting")
        
        # 限制样本数量
        if len(all_features) > max_samples:
            indices = np.random.choice(len(all_features), max_samples, replace=False)
            all_features = all_features[indices]
        
        print(f"Fitting PCA with {len(all_features)} samples")
        
        # 拟合PCA
        self.pca = PCA(n_components=self.pca_dim)
        self.pca.fit(all_features)
        self.pca_fitted = True
        
        # 保存PCA模型
        self.save_pca()
        
        # 打印PCA信息
        explained_variance_ratio = self.pca.explained_variance_ratio_
        print(f"PCA explained variance ratio: {explained_variance_ratio}")
        print(f"Total explained variance: {explained_variance_ratio.sum():.4f}")
        
        # 清空训练特征以节省内存
        self.training_features = []
    
    def save_pca(self):
        """
        保存PCA模型
        """
        if self.pca is not None:
            os.makedirs(os.path.dirname(self.pca_save_path), exist_ok=True)
            with open(self.pca_save_path, 'wb') as f:
                pickle.dump(self.pca, f)
            print(f"PCA model saved to {self.pca_save_path}")
    
    def load_pca(self):
        """
        加载PCA模型
        """
        if os.path.exists(self.pca_save_path):
            with open(self.pca_save_path, 'rb') as f:
                self.pca = pickle.load(f)
            self.pca_fitted = True
            print(f"PCA model loaded from {self.pca_save_path}")
            return True
        return False
    
    def forward(self, images):
        """
        前向传播
        
        Args:
            images (torch.Tensor): 输入图像，形状为 (B, C, H, W) 或 (B, T, C, H, W)
            
        Returns:
            torch.Tensor: PCA降维后的特征，形状为 (B, pca_dim) 或 (B, T, pca_dim)
        """
        original_shape = images.shape
        
        # 处理时序维度
        if len(images.shape) == 5:  # (B, T, C, H, W)
            B, T = images.shape[:2]
            images = images.view(-1, *images.shape[2:])  # (B*T, C, H, W)
        else:
            B, T = images.shape[0], 1
        
        # 预处理图像
        images = self._preprocess_images(images)
        
        # 提取CLIP特征
        clip_features = self._extract_clip_features(images)  # (B*T, clip_dim)
        
        # 如果在训练模式且PCA未拟合，收集特征
        if self.training and not self.pca_fitted:
            self.training_features.append(clip_features.detach().cpu().numpy())
            # 当收集到足够特征时，自动拟合PCA
            total_samples = sum(len(f) for f in self.training_features)
            if total_samples >= 1000:  # 收集1000个样本后拟合PCA
                print(f"Auto-fitting PCA with {total_samples} collected samples...")
                self.fit_pca()
        
        # 如果PCA未拟合，尝试加载或使用原始特征
        if not self.pca_fitted:
            if not self.load_pca():
                # 如果没有PCA模型，直接返回CLIP特征的前pca_dim维
                print("Warning: Using truncated CLIP features instead of PCA features")
                truncated_features = clip_features[:, :self.pca_dim].to(images.device)
                if len(original_shape) == 5:
                    return truncated_features.view(B, T, self.pca_dim)
                else:
                    return truncated_features
        
        # 应用PCA降维
        clip_features_np = clip_features.detach().cpu().numpy()
        pca_features = self.pca.transform(clip_features_np)
        pca_features = torch.from_numpy(pca_features).float().to(clip_features.device)
        
        # 恢复原始形状
        if len(original_shape) == 5:
            pca_features = pca_features.view(B, T, self.pca_dim)
        
        return pca_features
    
    def get_output_dim(self):
        """
        获取输出特征维度
        """
        return self.pca_dim


def get_clip_pca_encoder(clip_model_name='ViT-B/32', pca_dim=8, device='cuda', pca_save_path=None, **kwargs):
    """
    工厂函数，用于创建CLIP+PCA编码器
    
    Args:
        clip_model_name (str): CLIP模型名称
        pca_dim (int): PCA降维维度
        device (str): 设备
        pca_save_path (str): PCA保存路径
        **kwargs: 其他参数（兼容性）
        
    Returns:
        CLIPPCAEncoder: CLIP+PCA编码器实例
    """
    return CLIPPCAEncoder(
        clip_model_name=clip_model_name,
        pca_dim=pca_dim,
        device=device,
        pca_save_path=pca_save_path
    )