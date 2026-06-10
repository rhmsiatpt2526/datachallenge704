import torch


def weighted_mse_loss(y_pred, y):
    weights = 1 / 30 + y
    return torch.sum(weights * (y_pred - y) ** 2) / torch.sum(weights)


def weighted_mse_per_group(y_pred, y, mask):
    """
    Calcule la MSE pondérée sur un sous-groupe du batch.
    mask: bool tensor de shape (batch_size, 1) ou (batch_size,)
    """
    mask = mask.view(-1, 1)

    if mask.sum() == 0:
        return None

    y_pred_group = y_pred[mask]
    y_group = y[mask]

    weights = 1 / 30 + y_group
    return torch.sum(weights * (y_pred_group - y_group) ** 2) / torch.sum(weights)


def weighted_mse_gender_gap_loss(y_pred, y, gender, lambda_gap=0.1):
    """
    Loss alignée avec la métrique finale :
    - weighted MSE globale
    - pénalité si erreur male/female déséquilibrée

    Hypothèse :
    gender == 0.0 pour female
    gender == 1.0 pour male
    """
    base_loss = weighted_mse_loss(y_pred, y)

    gender = gender.view(-1, 1).float()

    female_mask = gender == 0.0
    male_mask = gender == 1.0

    female_loss = weighted_mse_per_group(y_pred, y, female_mask)
    male_loss = weighted_mse_per_group(y_pred, y, male_mask)

    # Si le batch ne contient qu'un seul genre, on ne peut pas calculer le gap.
    # Dans ce cas, on revient simplement à la loss classique.
    if female_loss is None or male_loss is None:
        return base_loss

    gender_gap = torch.abs(male_loss - female_loss)

    return base_loss + lambda_gap * gender_gap
