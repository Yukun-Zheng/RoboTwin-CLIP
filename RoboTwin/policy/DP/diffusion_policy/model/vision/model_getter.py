import torch
import torchvision
from .clip_pca_encoder import get_clip_pca_encoder


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    # resnet_new = torch.nn.Sequential(
    #     resnet,
    #     torch.nn.Linear(512, 128)
    # )
    # return resnet_new
    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model


def get_clip_pca(clip_model_name='ViT-B/32', pca_dim=8, device='cuda', **kwargs):
    """
    获取CLIP+PCA编码器
    
    Args:
        clip_model_name (str): CLIP模型名称，如'ViT-B/32', 'RN50'等
        pca_dim (int): PCA降维后的维度，默认为8
        device (str): 设备类型，默认为'cuda'
        **kwargs: 其他参数
    
    Returns:
        CLIPPCAEncoder: CLIP+PCA编码器实例
    """
    return get_clip_pca_encoder(
        clip_model_name=clip_model_name,
        pca_dim=pca_dim,
        device=device,
        **kwargs
    )