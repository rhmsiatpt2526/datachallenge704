import torch.nn as nn
from torchvision import models
from transformers import AutoModel


def make_regression_head(in_features):
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


class DinoV2Regressor(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", freeze_backbone=True):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.regressor = make_regression_head(hidden_size)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]
        return self.regressor(features)


class DinoV3Regressor(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.regressor = make_regression_head(hidden_size)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_last_n_blocks)

        def unfreeze_last_blocks(self, n):
            candidates = [
                "encoder.layer",
                "encoder.layers",
                "dinov3.encoder.layer",
                "dinov3.encoder.layers",
                "model.encoder.layer",
                "model.encoder.layers",
                "layers",
                "blocks",
            ]

            blocks = None

            for path in candidates:
                current = self.backbone
                ok = True

                for attr in path.split("."):
                    if hasattr(current, attr):
                        current = getattr(current, attr)
                    else:
                        ok = False
                        break

                if ok:
                    blocks = current
                    print(f"Found transformer blocks at: {path}")
                    break

            if blocks is None:
                print(self.backbone)
                raise AttributeError(
                    "Impossible de trouver les blocs Transformer du backbone DINOv3."
                )

            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True

            for name, module in self.backbone.named_modules():
                if name.endswith("layernorm") or name.endswith("norm"):
                    for param in module.parameters():
                        param.requires_grad = True

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]
        return self.regressor(features)


def build_model(model_name, freeze_backbone=True, unfreeze_last_n_blocks=0):
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

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
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

    elif model_name == "dinov2_small":
        return DinoV2Regressor(
            model_name="facebook/dinov2-small",
            freeze_backbone=freeze_backbone,
        )

    elif model_name == "dinov2_base":
        return DinoV2Regressor(
            model_name="facebook/dinov2-base",
            freeze_backbone=freeze_backbone,
        )

    elif model_name == "dinov3-vits16":
        return DinoV3Regressor(
            model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    elif model_name == "dinov3-vitb16":
        return DinoV3Regressor(
            model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = any(keyword in name for keyword in head_keywords)

    return model
