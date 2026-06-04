import torch


def weighted_mse_loss(y_pred, y):
    weights = 1 / 30 + y
    return torch.sum(weights * (y_pred - y) ** 2) / torch.sum(weights)
