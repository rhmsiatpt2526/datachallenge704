import torch
from tqdm.auto import tqdm

from src.losses import weighted_mse_loss


def train_one_epoch(
    model, loader, optimizer, device, epoch, num_epochs, use_non_blocking=False
):
    model.train()
    running_loss = 0.0
    seen_samples = 0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for batch_idx, (x, y, gender, filename) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=use_non_blocking)
        y = y.to(device, non_blocking=use_non_blocking).view(-1, 1)

        y_pred = model(x)
        loss = weighted_mse_loss(y_pred, y)

        if torch.isnan(loss):
            print("NaN detected in batch")
            print(filename)
            print("label", y)
            print("y_pred", y_pred)
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        seen_samples += batch_size
        running_loss += loss.item() * batch_size
        mean_loss = running_loss / max(seen_samples, 1)

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "mean_loss": f"{mean_loss:.4f}",
                "batch": batch_idx,
            }
        )

    epoch_loss = running_loss / max(seen_samples, 1)
    print(f"Epoch {epoch + 1}/{num_epochs} - mean loss: {epoch_loss:.4f}")

    return epoch_loss
