import torch.nn as nn
from torchvision import models
from transformers import AutoModel


def make_regression_head(in_features):
    return nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )


class DinoV3Regressor(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vits16", freeze_backbone=True):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]

        return self.regressor(features)


def build_model(model_name, freeze_backbone=True):
    if model_name == "mobilenetv3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = make_regression_head(in_features)
        head_keywords = ["classifier"]

    elif model_name == "mobilenetv3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = make_regression_head(in_features)
        head_keywords = ["classifier"]

    elif model_name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = make_regression_head(in_features)
        head_keywords = ["classifier"]

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = make_regression_head(in_features)
        head_keywords = ["fc"]

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = make_regression_head(in_features)
        head_keywords = ["classifier"]

    elif model_name == "dinov3-vits16":
        return DinoV3Regressor(
            model_name="facebook/dinov3-vits16", freeze_backbone=freeze_backbone
        )

    elif model_name == "dinov3-vitb16":
        return DinoV3Regressor(
            model_name="facebook/dinov3-vitb16", freeze_backbone=freeze_backbone
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = any(keyword in name for keyword in head_keywords)

    return model
