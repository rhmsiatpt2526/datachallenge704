import torch.nn as nn
from torchvision import models


def replace_classifier(model, in_features):
    return nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )


def build_model(model_name, freeze_backbone=True):
    if model_name == "mobilenetv3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = replace_classifier(model, in_features)

    elif model_name == "mobilenetv3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = replace_classifier(model, in_features)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    return model
