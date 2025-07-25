import torch
import torchvision
try:
    import clip
except ImportError:
    clip = None
    print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")


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


def get_clip(name="ViT-B/32", freeze=True, feature_layer=-1, **kwargs):
    """
    Get CLIP vision encoder
    name: CLIP model name like "ViT-B/32", "ViT-L/14", "RN50", etc.
    freeze: whether to freeze CLIP parameters
    feature_layer: which layer to extract features from (-1 for final, -2 for second last, etc.)
    """
    if clip is None:
        raise ImportError("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    
    device = "cpu"  # Load on CPU first, will be moved to GPU later
    model, preprocess = clip.load(name, device=device)
    
    class CLIPVisionEncoder(torch.nn.Module):
        def __init__(self, clip_model, freeze=True, feature_layer=-1, model_name=""):
            super().__init__()
            self.visual = clip_model.visual
            self.feature_layer = feature_layer
            self.name = model_name  # Store model name for size detection
            
            if freeze:
                for param in self.visual.parameters():
                    param.requires_grad = False
                    
            # Get output dimension
            if hasattr(self.visual, 'output_dim'):
                self.output_dim = self.visual.output_dim
            elif hasattr(self.visual, 'embed_dim'):
                self.output_dim = self.visual.embed_dim
            else:
                # For different CLIP models
                if "RN50" in name:
                    self.output_dim = 1024
                elif "ViT-L" in name:
                    self.output_dim = 768
                else:
                    self.output_dim = 512  # ViT-B default
                
        def forward(self, x):
            # x shape: (batch_size, 3, 224, 224)
            if hasattr(self.visual, 'conv1'):  # ResNet-based
                return self.visual(x)
            else:  # ViT-based
                x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.visual.positional_embedding.to(x.dtype)
                x = self.visual.ln_pre(x)
                
                x = x.permute(1, 0, 2)  # NLD -> LND
                
                # Extract features from specific layer
                if self.feature_layer == -1:
                    x = self.visual.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.visual.ln_post(x[:, 0, :])
                else:
                    # Extract from intermediate layer
                    for i, layer in enumerate(self.visual.transformer.resblocks):
                        x = layer(x)
                        if i == len(self.visual.transformer.resblocks) + self.feature_layer:
                            break
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = x[:, 0, :]  # Take CLS token
                    
                return x
    
    encoder = CLIPVisionEncoder(model, freeze=freeze, feature_layer=feature_layer, model_name=name)
    return encoder
