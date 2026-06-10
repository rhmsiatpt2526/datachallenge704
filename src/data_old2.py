import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from src.config import TRAIN_CSV, TEST_CSV, IMAGE_DIR
from src.dataset import OcclusionDataset


def load_dataframes():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    df_train = df_train.dropna(subset=["filename", "FaceOcclusion", "gender"])
    df_test = df_test.dropna(subset=["filename"])

    return df_train, df_test


def make_train_val_split(df_train, test_size=0.2, random_state=42):
    df_train = df_train.copy()

    df_train["occ_bin"] = pd.cut(
        df_train["FaceOcclusion"],
        bins=[-0.001, 0.05, 0.10, 0.20, 0.35, 1.0],
        labels=["0-5", "5-10", "10-20", "20-35", "35+"],
    )

    df_train["stratify_col"] = (
        df_train["gender"].astype(str) + "_" + df_train["occ_bin"].astype(str)
    )

    df_train, df_val = train_test_split(
        df_train,
        test_size=test_size,
        random_state=random_state,
        stratify=df_train["stratify_col"],
    )

    df_train = df_train.drop(columns=["occ_bin", "stratify_col"]).reset_index(drop=True)
    df_val = df_val.drop(columns=["occ_bin", "stratify_col"]).reset_index(drop=True)

    return df_train, df_val


def create_dataloaders(args, use_pin_memory=False):
    df_train, df_test = load_dataframes()
    df_train, df_val = make_train_val_split(df_train)

    training_set = OcclusionDataset(df_train, IMAGE_DIR, training=True)
    validation_set = OcclusionDataset(df_val, IMAGE_DIR, training=False)
    test_set = OcclusionDataset(df_test, IMAGE_DIR, training=False)

    params_train = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": use_pin_memory,
    }

    params_val = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": use_pin_memory,
    }

    training_loader = torch.utils.data.DataLoader(training_set, **params_train)
    validation_loader = torch.utils.data.DataLoader(validation_set, **params_val)
    test_loader = torch.utils.data.DataLoader(test_set, **params_val)

    return training_loader, validation_loader, test_loader
