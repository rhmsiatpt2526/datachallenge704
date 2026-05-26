#!/usr/bin/env python
# coding: utf-8

# # Baseline model - MobileNetV3-Small

# In[3]:


from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision import transforms
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")
if torch.cuda.is_available():
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
# print(f"Nom du GPU : {torch.device_name(0)}")

# import torch_directml
# device = torch_directml.device()
# print(f"Device utilisé : {device}")
# print(f"Nom du GPU : {torch_directml.device_name(0)}")


# In[2]:


df_train = pd.read_csv("../occlusion_datasets/train.csv", delimiter=",")
df_test = pd.read_csv("../occlusion_datasets/test_students.csv", delimiter=",")

image_dir = "../crops/Crop_224_5fp_100K"

df_train = df_train.dropna()
df_test = df_test.dropna()

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


# In[3]:


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
        img = Image.open(f"{self.image_dir}/{filename}").convert("RGB")

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


# In[4]:


training_set = Dataset(df_train, image_dir)
validation_set = Dataset(df_val, image_dir)
test_set = Dataset(df_test, image_dir, training=False)

params_train = {"batch_size": 128, "shuffle": True, "num_workers": 0}

params_val = {"batch_size": 128, "shuffle": False, "num_workers": 0}

training_generator = torch.utils.data.DataLoader(training_set, **params_train)
validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
test_generator = torch.utils.data.DataLoader(test_set, **params_val)


# In[ ]:


from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

weights = MobileNet_V3_Small_Weights.DEFAULT
# model = torchvision.models.mobilenet_v3_small(num_classes=1, weights=weights)
model = mobilenet_v3_small(weights=weights)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)


def weighted_mse_loss(y_pred, y):
    weights = 1 / 30 + y
    return torch.mean(weights * (y_pred - y) ** 2)


# loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    # , foreach=False quand sur torch directml sinon gpu cluster
)
model = model.to(device)


# In[6]:


num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    seen_samples = 0

    pbar = tqdm(
        training_generator,
        total=len(training_generator),
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for batch_idx, (X, y, gender, filename) in enumerate(pbar, start=1):
        X, y = X.to(device), y.to(device)
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


# In[7]:


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


results_list = []
model.eval()
with torch.no_grad():
    for batch_idx, (X, y, gender, filename) in tqdm(
        enumerate(validation_generator), total=len(validation_generator)
    ):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        for i in range(len(X)):
            results_list.append(
                {
                    "filename": filename[i],
                    "pred": float(torch.clamp(y_pred[i], 0, 1).item()),
                    "target": float(y[i]),
                    "gender": float(gender[i]),
                }
            )


# In[8]:


results_df = pd.DataFrame(results_list)
results_male = results_df.loc[results_df["gender"] == 1.0]
results_female = results_df.loc[results_df["gender"] == 0.0]
metric_fn(results_male, results_female)


# In[9]:


results_list = []
model.eval()
with torch.no_grad():
    for batch_idx, (X, filename) in tqdm(
        enumerate(test_generator), total=len(test_generator)
    ):
        X = X.to(device)
        y_pred = model(X)

        for i in range(len(X)):
            results_list.append(
                {
                    "filename": filename[i],
                    "FaceOcclusion": float(torch.clamp(y_pred[i], 0, 1).item()),
                }
            )
results_df = pd.DataFrame(results_list)


# In[10]:


from datetime import datetime

MODEL_NAME = "mobilenetv3_small"
RUNS_DIR = Path("../logs") / MODEL_NAME
SUBMISSIONS_DIR = Path("../submissions")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def next_run_id(base_dir, model_name):
    run_ids = []
    for path in base_dir.glob(f"{model_name}_run*.md"):
        suffix = path.stem.replace(f"{model_name}_run", "")
        if suffix.isdigit():
            run_ids.append(int(suffix))
    return max(run_ids, default=0) + 1


def collect_predictions(loader, split_name):
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
            X, y = X.to(device), y.to(device)
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


run_id = next_run_id(RUNS_DIR, MODEL_NAME)
run_tag = f"{MODEL_NAME}_run{run_id:03d}"
timestamp = datetime.now().isoformat(timespec="seconds")

train_results_df = collect_predictions(training_generator, "train")
val_results_df = collect_predictions(validation_generator, "validation")
train_stats = split_errors(train_results_df)
val_stats = split_errors(val_results_df)

submission_df = results_df.copy()
submission_df["FaceOcclusion"] = submission_df["FaceOcclusion"].clip(0, 1)
submission_df["gender"] = "x"

submission_path = SUBMISSIONS_DIR / f"{run_tag}.csv"
submission_df.to_csv(submission_path, sep=",", index=False)

log_lines = [
    f"run: {run_tag}",
    f"timestamp: {timestamp}",
    f"model: {MODEL_NAME}",
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
