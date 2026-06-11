import pandas as pd
import torch
from tqdm.auto import tqdm


def predict_with_optional_tta(model, x, tta=False):
    """
    Predict on a batch.

    If tta=True, average predictions from:
    - the original batch
    - the horizontally flipped batch
    """
    y_pred = model(x)

    if tta:
        x_flip = torch.flip(x, dims=[3])  # BCHW -> flip width axis
        y_pred_flip = model(x_flip)
        y_pred = 0.5 * (y_pred + y_pred_flip)

    return y_pred


def collect_predictions(
    model,
    loader,
    split_name,
    device,
    use_non_blocking=False,
    tta=False,
):
    rows = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            loader, total=len(loader), desc=f"Predicting {split_name}", leave=False
        ):
            x, y, gender, filename = batch

            x = x.to(device, non_blocking=use_non_blocking)
            y = y.to(device, non_blocking=use_non_blocking)

            y_pred = predict_with_optional_tta(model, x, tta=tta)

            for i in range(len(x)):
                rows.append(
                    {
                        "filename": filename[i],
                        "pred": float(torch.clamp(y_pred[i], 0, 1).item()),
                        "target": float(y[i]),
                        "gender": float(gender[i]),
                    }
                )

    return pd.DataFrame(rows)


def predict_test(model, loader, device, use_non_blocking=False, tta=False):
    rows = []
    model.eval()

    with torch.no_grad():
        for x, filename in tqdm(loader, total=len(loader), desc="Predicting test"):
            x = x.to(device, non_blocking=use_non_blocking)
            y_pred = predict_with_optional_tta(model, x, tta=tta)

            for i in range(len(x)):
                rows.append(
                    {
                        "filename": filename[i],
                        "FaceOcclusion": float(torch.clamp(y_pred[i], 0, 1).item()),
                    }
                )

    return pd.DataFrame(rows)
