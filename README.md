# README — Final Ensemble for Data Challenge 704

## Summary

Final selected submission:

```text
ensemble_final_bestguess_no_run001_run002_mesogip.csv
```

Observed public leaderboard score:

```text
0.00109
```

This final model is a weighted ensemble of 6 `dinov3-vitb16` submission files. It does **not** use `run001`, and it explicitly includes the older `dinov3-vitb16_run002.csv`, which generalized better on the public leaderboard than some locally stronger single models.

---

## Exact ensemble composition

Formula:

```text
FaceOcclusion_final =
  0.21 * FaceOcclusion_18142
+ 0.18 * FaceOcclusion_18141
+ 0.17 * FaceOcclusion_18127
+ 0.24 * FaceOcclusion_run002
+ 0.14 * FaceOcclusion_18125
+ 0.06 * FaceOcclusion_18138
```

| Weight | Source | CSV file |
|---:|---|---|
| 0.21 | L40S seed777, u4, lr3e-5, gap0.02 | `submissions/dinov3-vitb16_job18142_seed777_mesogip_l40s_vitb16_u4_lr3e5_gap002_seed777_e20.csv` |
| 0.18 | L40S seed777, u4, lr2e-5, gap0.05 | `submissions/dinov3-vitb16_job18141_seed777_mesogip_l40s_vitb16_u4_lr2e5_gap005_seed777_e24.csv` |
| 0.17 | L40S seed777, u4, lr3e-5, gap0.05 | `submissions/dinov3-vitb16_job18127_seed777_mesogip_l40s_vitb16_u4_lr3e5_gap005_seed777_e20.csv` |
| 0.24 | Older DINOv3 `run002` | `submissions/dinov3-vitb16_run002.csv` |
| 0.14 | H100 seed123, full fine-tuning, lr8e-6, gap0.05 | `submissions/dinov3-vitb16_job18125_seed123_mesogip_h100_vitb16_full_lr8e6_gap005_seed123.csv` |
| 0.06 | L40S seed123, u4, lr3e-5, gap0.05 | `submissions/dinov3-vitb16_job18138_seed123_mesogip_l40s_vitb16_u4_lr3e5_gap005_seed123_e20.csv` |

---

## Constituent run identity cards

| ID | GPU | Seed | Epochs | Batch | LR | WD | Scheduler | min_lr | Freeze | Unfreeze | Gap loss | lambda_gap | TTA | Trainable params | Best epoch | Final val balanced metric |
|---:|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|---|---:|---:|---:|
| 18142 | L40S | 777 | 20 | 24 | 3e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.02 | True | 28,594,561 | 10 | 0.001061 |
| 18141 | L40S | 777 | 24 | 24 | 2e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.05 | True | 28,594,561 | 23 | 0.001063 |
| 18127 | L40S | 777 | 20 | 24 | 3e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.05 | True | 28,594,561 | 14 | 0.001063 |
| run002 | historical run | not recorded | 50 | 64 | 5e-6 | 1e-4 | cosine | 5e-7 | not explicit | likely 4* | not explicit | not explicit | not explicit | 28,594,561 | not recorded | 0.001616 |
| 18125 | H100 NVL | 123 | 20 | 16 | 8e-6 | 5e-5 | cosine | 1e-7 | False | 0 | True | 0.05 | True | 85,874,305 | 20 | 0.001138 |
| 18138 | L40S | 123 | 20 | 24 | 3e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.05 | True | 28,594,561 | 6 | 0.001157 |

\* For `run002`, the available metadata does not explicitly list `freeze_backbone` or `unfreeze_last_n_blocks`. However, its number of trainable parameters, `28,594,561`, matches the DINOv3 ViT-B/16 partial fine-tuning setup with 4 unfrozen blocks. To reproduce the final ensemble exactly, the required artifact is the CSV file `submissions/dinov3-vitb16_run002.csv`.

---

## Exact reproduction commands for Mesogip runs

From the project root:

```bash
cd ~/datachallenge704
source .venv_dinov3/bin/activate
```

### Run 18142

```bash
sbatch scripts/mesogip_l40s.sh \
  --model dinov3-vitb16 \
  --epochs 20 \
  --batch-size 24 \
  --lr 3e-5 \
  --weight-decay 5e-5 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 1e-7 \
  --val-every 1 \
  --seed 777 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 4 \
  --use-gender-gap-loss \
  --lambda-gap 0.02 \
  --tta \
  --experiment-name mesogip_l40s_vitb16_u4_lr3e5_gap002_seed777_e20
```

### Run 18141

```bash
sbatch scripts/mesogip_l40s.sh \
  --model dinov3-vitb16 \
  --epochs 24 \
  --batch-size 24 \
  --lr 2e-5 \
  --weight-decay 5e-5 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 1e-7 \
  --val-every 1 \
  --seed 777 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 4 \
  --use-gender-gap-loss \
  --lambda-gap 0.05 \
  --tta \
  --experiment-name mesogip_l40s_vitb16_u4_lr2e5_gap005_seed777_e24
```

### Run 18127

```bash
sbatch scripts/mesogip_l40s.sh \
  --model dinov3-vitb16 \
  --epochs 20 \
  --batch-size 24 \
  --lr 3e-5 \
  --weight-decay 5e-5 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 1e-7 \
  --val-every 1 \
  --seed 777 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 4 \
  --use-gender-gap-loss \
  --lambda-gap 0.05 \
  --tta \
  --experiment-name mesogip_l40s_vitb16_u4_lr3e5_gap005_seed777_e20
```

### Run 18125

```bash
sbatch scripts/mesogip.sh \
  --model dinov3-vitb16 \
  --epochs 20 \
  --batch-size 16 \
  --lr 8e-6 \
  --weight-decay 5e-5 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 1e-7 \
  --val-every 1 \
  --seed 123 \
  --use-gender-gap-loss \
  --lambda-gap 0.05 \
  --tta \
  --experiment-name mesogip_h100_vitb16_full_lr8e6_gap005_seed123
```

### Run 18138

```bash
sbatch scripts/mesogip_l40s.sh \
  --model dinov3-vitb16 \
  --epochs 20 \
  --batch-size 24 \
  --lr 3e-5 \
  --weight-decay 5e-5 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 1e-7 \
  --val-every 1 \
  --seed 123 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 4 \
  --use-gender-gap-loss \
  --lambda-gap 0.05 \
  --tta \
  --experiment-name mesogip_l40s_vitb16_u4_lr3e5_gap005_seed123_e20
```

### Historical run002

This run is used directly through its CSV:

```text
submissions/dinov3-vitb16_run002.csv
```

Available metadata:

```text
model: dinov3-vitb16
epochs: 50
batch_size: 64
lr: 5e-6
weight_decay: 1e-4
scheduler: cosine
min_lr: 5e-7
num_workers: 8
val_every: 5
total_params: 85,874,305
trainable_params: 28,594,561
checkpoint_file: dinov3-vitb16_run002.pt
submission_file: dinov3-vitb16_run002.csv
validation_balanced_metric: 0.001616
```

---

## Exact ensemble construction command

```bash
cd ~/datachallenge704
source .venv_dinov3/bin/activate

python - <<'PY'
from pathlib import Path
import pandas as pd

sub_dir = Path("submissions")

paths = {
    "s777_gap002": sub_dir / "dinov3-vitb16_job18142_seed777_mesogip_l40s_vitb16_u4_lr3e5_gap002_seed777_e20.csv",
    "s777_lr2": sub_dir / "dinov3-vitb16_job18141_seed777_mesogip_l40s_vitb16_u4_lr2e5_gap005_seed777_e24.csv",
    "s777_gap005": sub_dir / "dinov3-vitb16_job18127_seed777_mesogip_l40s_vitb16_u4_lr3e5_gap005_seed777_e20.csv",
    "old_run002": sub_dir / "dinov3-vitb16_run002.csv",
    "h100_full": sub_dir / "dinov3-vitb16_job18125_seed123_mesogip_h100_vitb16_full_lr8e6_gap005_seed123.csv",
    "l40s_seed123": sub_dir / "dinov3-vitb16_job18138_seed123_mesogip_l40s_vitb16_u4_lr3e5_gap005_seed123_e20.csv",
}

names = [
    "s777_gap002",
    "s777_lr2",
    "s777_gap005",
    "old_run002",
    "h100_full",
    "l40s_seed123",
]

weights = [0.21, 0.18, 0.17, 0.24, 0.14, 0.06]

dfs = []
for name in names:
    path = paths[name]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "FaceOcclusion" not in df.columns:
        raise ValueError(f"{path} has no FaceOcclusion column")
    print(name, path.name, df.shape)
    dfs.append(df)

base = dfs[0].copy()

if "ID" in base.columns:
    for name, df in zip(names[1:], dfs[1:]):
        if "ID" not in df.columns:
            raise ValueError(f"{name} has no ID column")
        if not base["ID"].equals(df["ID"]):
            raise ValueError(f"ID mismatch for {name}")
    print("Alignment OK using ID")
else:
    for name, df in zip(names[1:], dfs[1:]):
        if len(df) != len(base):
            raise ValueError(f"Row mismatch for {name}")
    print("Alignment OK using row order")

weights = [w / sum(weights) for w in weights]

pred = 0
for name, w, df in zip(names, weights, dfs):
    print(f"{name:15s} weight={w:.3f}")
    pred += w * df["FaceOcclusion"]

base["FaceOcclusion"] = pred.clip(0, 1)

if "gender" in base.columns:
    base["gender"] = "x"

out = sub_dir / "ensemble_final_bestguess_no_run001_run002_mesogip.csv"
base.to_csv(out, index=False)

print("Saved:", out)
print(base.head())
PY
```

---

## Verification and download

```bash
ls -lh submissions/ensemble_final_bestguess_no_run001_run002_mesogip.csv
head -5 submissions/ensemble_final_bestguess_no_run001_run002_mesogip.csv
```

From a local Git Bash terminal:

```bash
scp mesogip:~/datachallenge704/submissions/ensemble_final_bestguess_no_run001_run002_mesogip.csv .
```

---

## Reproducibility notes

- All ensemble components predict the `FaceOcclusion` column.
- Files are aligned by `ID` if present, otherwise by row order.
- The final prediction is clipped to `[0, 1]`.
- If a `gender` column exists in the submission file, it is set to `x` to preserve the expected submission format.
- The final model is a **post-hoc CSV ensemble**, not a single PyTorch checkpoint.
- Hyperparameters, checkpoints, submission files and Markdown run logs were saved for reproducibility.

---

## Main limitations and possible improvements

The final solution was built iteratively rather than through a fully systematic hyperparameter search. The main limitations are:

- no rigorous Optuna-style hyperparameter optimization;
- limited number of validation splits;
- no final K-fold ensemble;
- ensembling weights selected empirically using validation and public leaderboard feedback;
- limited exploration of alternative strong backbones;
- no pseudo-labeling or synthetic occlusion generation;
- limited handling of distribution shift between validation and public test data.

The most promising next steps would be:

1. train a K-fold DINOv3 ensemble;
2. add a truly different backbone for ensemble diversity;
3. run a full fine-tuning DINOv3 seed777 model;
4. optimize the gender-gap penalty more systematically;
5. explore pseudo-labeling or better calibration under distribution shift.

---

---

## Project overview

The goal of Data Challenge 704 was to predict a continuous face occlusion score from cropped face images:

```text
FaceOcclusion ∈ [0, 1]
```

The task was treated as supervised image regression. The final pipeline supports pretrained backbones, a regression head with sigmoid output, gender-aware validation, best-checkpoint selection, Test-Time Augmentation, multi-seed training, MLflow logging, Slurm execution, and post-hoc ensembling.

The final performance was obtained through a progressive and iterative process:

1. Start from lightweight CNN baselines.
2. Move to stronger pretrained backbones.
3. Fine-tune DINOv3 ViT-B/16 with controlled backbone unfreezing.
4. Add gender-gap-aware loss and validation.
5. Run several seeds and fine-tuning variants on GPU clusters.
6. Build a weighted ensemble to improve public leaderboard generalization.

---

## Data and evaluation

Input files are stored in:

```text
occlusion_datasets/train.csv
occlusion_datasets/test_students.csv
crops/Crop_224_5fp_100K/
```

Useful dataset facts:

- `train.csv`: 100,000 rows.
- `test_students.csv`: 29,980 rows.
- Images: 224×224 RGB `.webp` crops.
- Main columns: `filename`, `FaceOcclusion`, `gender`.

The validation metric used in the project was gender-balanced:

```python
metric = (error_male + error_female) / 2 + abs(error_male - error_female)
```

This encouraged both low global error and similar error levels across gender groups.

---

## Modeling path

Several model families were tested before converging to DINOv3:

```text
mobilenetv3_small
mobilenetv3_large
efficientnet_b0
efficientnet_b1
resnet50
convnext_tiny
dinov2_small
dinov2_base
dinov3-vits16
dinov3-vitb16
```

The final backbone was:

```text
dinov3-vitb16
```

This choice was driven by the quality of its pretrained visual representations, its capacity, and the empirical validation and leaderboard results.

The regression head evolved from a simple one-layer head to a deeper MLP-style head ending with a sigmoid, so that predictions remain in `[0, 1]`.

---

## Training strategy

### Fine-tuning

Training first used a frozen backbone and trained only the regression head. Later runs used partial fine-tuning by unfreezing the last transformer blocks:

```bash
--freeze-backbone --unfreeze-last-n-blocks 4
```

The best final ensemble mostly relies on DINOv3 ViT-B/16 runs with 4 unfrozen blocks, plus one full fine-tuning run (`18125`) to add diversity.

### Data augmentation

The training pipeline used lightweight augmentations such as:

- horizontal flip;
- color jitter;
- affine transformations;
- normalization;
- random erasing.

### Loss

The base loss is a weighted MSE that gives more importance to larger occlusion values:

```python
weights = 1 / 30 + y
loss = torch.sum(weights * (y_pred - y) ** 2) / torch.sum(weights)
```

A gender-gap penalty can be added:

```python
loss = weighted_mse + lambda_gap * abs(male_loss - female_loss)
```

The best values found empirically were around:

```text
lambda_gap = 0.02 to 0.05
```

### Learning rate and regularization

The best runs used small learning rates with cosine scheduling:

```bash
--scheduler cosine --min-lr 1e-7
```

Weight decay was used to improve regularization, typically between `5e-5` and `1e-4`.

### Test-Time Augmentation

TTA was enabled for final validation and test predictions:

```bash
--tta
```

The implemented TTA averages predictions on the original image and its horizontally flipped version.

### Splitting and seeds

The train/validation split was stratified using both occlusion level and gender. Multiple seeds were tested. Some seeds gave much better local validation scores, but the best local single model did not always generalize best on the public leaderboard. This mismatch motivated the final weighted ensemble.

---

## Compute setup

The work was run progressively on several environments:

- local GPU for early baselines;
- Télécom Paris cluster, mainly P100 GPUs;
- external H100 RunPod experiments;
- Mesogip / ENSTA cluster, using H100 NVL and L40S GPUs for the final runs.

Most final models were trained on Mesogip using Slurm scripts:

```text
scripts/mesogip.sh        # H100 jobs
scripts/mesogip_l40s.sh   # L40S jobs
```

Run logging was improved during the project to avoid checkpoint collisions between parallel Slurm jobs and to make each run reproducible through explicit experiment names, seeds, logs, checkpoints, and submission files.
