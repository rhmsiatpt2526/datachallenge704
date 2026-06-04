#!/usr/bin/env python
# coding: utf-8

# # Baseline model - MobileNetV3-Small

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm.auto import tqdm
from datetime import datetime
import argparse


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "occlusion_datasets" / "train.csv"
TEST_CSV = ROOT / "occlusion_datasets" / "test_students.csv"
IMAGE_DIR = ROOT / "crops" / "Crop_224_5fp_100K"

MODEL_NAME = "mobilenetv3_small"


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, df, image_dir, training=True):
        "Initialization"
        self.training = training
        self.image_dir = image_dir
        self.df = df
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.df)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        row = self.df.loc[index]
        filename = row["filename"]

        # Load data and get label
        img = Image.open(self.image_dir / filename).convert("RGB")

        X = self.transform(img)

        if self.training:
            y = row["FaceOcclusion"]
            y = np.float32(y)
            gender = row["gender"]
            return X, y, gender, filename
        else:
            y = None
            gender = None
            return X, filename


def weighted_mse_loss(y_pred, y):
    weights = 1 / 30 + y
    return torch.sum(weights * (y_pred - y) ** 2) / torch.sum(weights)


def error_fn(df):
    pred = df.loc[:, "pred"]
    pred = np.clip(pred, 0, 1)
    ground_truth = df.loc[:, "target"]
    weight = 1 / 30 + ground_truth
    return np.sum(((pred - ground_truth) ** 2) * weight, axis=0) / np.sum(
        weight, axis=0
    )


def metric_fn(female, male):
    err_male = error_fn(male)
    err_female = error_fn(female)
    return (err_male + err_female) / 2 + abs(err_male - err_female)


def next_run_id(base_dir, model_name):
    run_ids = []
    for path in base_dir.glob(f"{model_name}_run*.md"):
        suffix = path.stem.replace(f"{model_name}_run", "")
        if suffix.isdigit():
            run_ids.append(int(suffix))
    return max(run_ids, default=0) + 1


def collect_predictions(model, loader, split_name, device, use_non_blocking=False):
    rows = []
    model.eval()
    with torch.no_grad():
        for _, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Predicting {split_name}",
            leave=False,
        ):
            X, y, gender, filename = batch
            X = X.to(device, non_blocking=use_non_blocking)
            y = y.to(device, non_blocking=use_non_blocking)

            y_pred = model(X)

            for i in range(len(X)):
                rows.append(
                    {
                        "filename": filename[i],
                        "pred": float(torch.clamp(y_pred[i], 0, 1).item()),
                        "target": float(y[i]),
                        "gender": float(gender[i]),
                    }
                )

    return pd.DataFrame(rows)


def split_errors(df):
    female = df.loc[df["gender"] == 0.0]
    male = df.loc[df["gender"] == 1.0]
    female_error = error_fn(female)
    male_error = error_fn(male)
    return {
        "error": error_fn(df),
        "female_error": female_error,
        "male_error": male_error,
        "gender_gap": abs(male_error - female_error),
        "balanced_metric": metric_fn(female, male),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    if torch.cuda.is_available():
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

    RUNS_DIR = ROOT / "logs" / MODEL_NAME
    SUBMISSIONS_DIR = ROOT / "submissions"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR = ROOT / "checkpoints" / MODEL_NAME
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    df_train = df_train.dropna(subset=["filename", "FaceOcclusion", "gender"])
    df_test = df_test.dropna(subset=["filename"])

    df_train["occ_bin"] = pd.cut(
        df_train["FaceOcclusion"],
        bins=[-0.001, 0.05, 0.10, 0.20, 0.35, 1.0],
        labels=["0-5", "5-10", "10-20", "20-35", "35+"],
    )

    df_train["stratify_col"] = (
        df_train["gender"].astype(str) + "_" + df_train["occ_bin"].astype(str)
    )

    df_train, df_val = train_test_split(
        df_train, test_size=0.2, random_state=42, stratify=df_train["stratify_col"]
    )

    df_train = df_train.drop(columns=["occ_bin", "stratify_col"]).reset_index(drop=True)
    df_val = df_val.drop(columns=["occ_bin", "stratify_col"]).reset_index(drop=True)

    training_set = Dataset(df_train, IMAGE_DIR)
    validation_set = Dataset(df_val, IMAGE_DIR)
    test_set = Dataset(df_test, IMAGE_DIR, training=False)

    params_train = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    params_val = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    training_loader = torch.utils.data.DataLoader(training_set, **params_train)
    validation_loader = torch.utils.data.DataLoader(validation_set, **params_val)
    test_loader = torch.utils.data.DataLoader(test_set, **params_val)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model = model.to(device)
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        pbar = tqdm(
            training_loader,
            total=len(training_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
        )

        for batch_idx, (X, y, gender, filename) in enumerate(pbar, start=1):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y = y.view(-1, 1)

            y_pred = model(X)
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

            batch_size = X.size(0)
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

        print(
            f"Epoch {epoch + 1}/{num_epochs} - mean loss: {running_loss / max(seen_samples, 1):.4f}"
        )

    results_list_test = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, filename) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            X = X.to(device, non_blocking=True)
            y_pred = model(X)

            for i in range(len(X)):
                results_list_test.append(
                    {
                        "filename": filename[i],
                        "FaceOcclusion": float(torch.clamp(y_pred[i], 0, 1).item()),
                    }
                )
    results_df_test = pd.DataFrame(results_list_test)

    run_id = next_run_id(RUNS_DIR, MODEL_NAME)
    run_tag = f"{MODEL_NAME}_run{run_id:03d}"
    timestamp = datetime.now().isoformat(timespec="seconds")

    train_results_df = collect_predictions(
        model, training_loader, "train", device, use_non_blocking=False
    )

    val_results_df = collect_predictions(
        model, validation_loader, "validation", device, use_non_blocking=False
    )
    train_stats = split_errors(train_results_df)
    val_stats = split_errors(val_results_df)

    checkpoint_path = CHECKPOINTS_DIR / f"{run_tag}.pt"

    torch.save(
        {
            "run_tag": run_tag,
            "model_name": MODEL_NAME,
            "epochs": num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_balanced_metric": val_stats["balanced_metric"],
            "validation_error": val_stats["error"],
            "train_error": train_stats["error"],
        },
        checkpoint_path,
    )

    submission_df = results_df_test.copy()
    submission_df["FaceOcclusion"] = submission_df["FaceOcclusion"].clip(0, 1)
    submission_df["gender"] = "x"

    submission_path = SUBMISSIONS_DIR / f"{run_tag}.csv"
    submission_df.to_csv(submission_path, sep=",", index=False)

    log_lines = [
        f"run: {run_tag}",
        f"timestamp: {timestamp}",
        f"model: {MODEL_NAME}",
        f"epochs: {num_epochs}",
        f"batch_size: {args.batch_size}",
        f"lr: {args.lr}",
        f"weight_decay: {args.weight_decay}",
        f"num_workers: {args.num_workers}",
        f"checkpoint_file: {checkpoint_path.name}",
        f"train_error: {train_stats['error']:.6f}",
        f"train_female_error: {train_stats['female_error']:.6f}",
        f"train_male_error: {train_stats['male_error']:.6f}",
        f"train_gender_gap: {train_stats['gender_gap']:.6f}",
        f"validation_error: {val_stats['error']:.6f}",
        f"validation_female_error: {val_stats['female_error']:.6f}",
        f"validation_male_error: {val_stats['male_error']:.6f}",
        f"validation_gender_gap: {val_stats['gender_gap']:.6f}",
        f"validation_balanced_metric: {val_stats['balanced_metric']:.6f}",
        "competition_test_error: NA (labels unavailable)",
        f"submission_file: {submission_path.name}",
        f"train_rows: {len(train_results_df)}",
        f"validation_rows: {len(val_results_df)}",
        f"test_rows: {len(submission_df)}",
    ]

    log_path = RUNS_DIR / f"{run_tag}.md"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Submission saved to: {submission_path}")
    print(f"Run log saved to: {log_path}")
    print(f"Validation balanced metric: {val_stats['balanced_metric']:.6f}")


if __name__ == "__main__":
    main()
