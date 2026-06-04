import numpy as np


def error_fn(df):
    pred = np.clip(df["pred"], 0, 1)
    target = df["target"]
    weight = 1 / 30 + target

    return np.sum(((pred - target) ** 2) * weight) / np.sum(weight)


def metric_fn(female, male):
    err_male = error_fn(male)
    err_female = error_fn(female)

    return (err_male + err_female) / 2 + abs(err_male - err_female)


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
