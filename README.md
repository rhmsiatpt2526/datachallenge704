# README — Ensemble final Data Challenge 704

## Résumé

Modèle final retenu :

```text
ensemble_final_bestguess_no_run001_run002_mesogip.csv
```

Score public leaderboard observé :

```text
0.00109
```

L'ensemble est une moyenne pondérée de 6 fichiers de soumission `dinov3-vitb16`. Il **n'utilise pas `run001`** et utilise explicitement l'ancien `dinov3-vitb16_run002.csv`, qui généralisait mieux sur le public leaderboard.

---

## Composition exacte de l'ensemble

Formule :

```text
FaceOcclusion_final =
  0.21 * FaceOcclusion_18142
+ 0.18 * FaceOcclusion_18141
+ 0.17 * FaceOcclusion_18127
+ 0.24 * FaceOcclusion_run002
+ 0.14 * FaceOcclusion_18125
+ 0.06 * FaceOcclusion_18138
```

| Poids | Source | Fichier CSV |
|---:|---|---|
| 0.21 | L40S seed777, u4, lr3e-5, gap0.02 | `submissions/dinov3-vitb16_job18142_seed777_mesogip_l40s_vitb16_u4_lr3e5_gap002_seed777_e20.csv` |
| 0.18 | L40S seed777, u4, lr2e-5, gap0.05 | `submissions/dinov3-vitb16_job18141_seed777_mesogip_l40s_vitb16_u4_lr2e5_gap005_seed777_e24.csv` |
| 0.17 | L40S seed777, u4, lr3e-5, gap0.05 | `submissions/dinov3-vitb16_job18127_seed777_mesogip_l40s_vitb16_u4_lr3e5_gap005_seed777_e20.csv` |
| 0.24 | Ancien run DINOv3 `run002` | `submissions/dinov3-vitb16_run002.csv` |
| 0.14 | H100 seed123, full fine-tuning, lr8e-6, gap0.05 | `submissions/dinov3-vitb16_job18125_seed123_mesogip_h100_vitb16_full_lr8e6_gap005_seed123.csv` |
| 0.06 | L40S seed123, u4, lr3e-5, gap0.05 | `submissions/dinov3-vitb16_job18138_seed123_mesogip_l40s_vitb16_u4_lr3e5_gap005_seed123_e20.csv` |

---

## Carte d'identité des runs constitutifs

| ID | GPU | Seed | Epochs | Batch | LR | WD | Scheduler | min_lr | Freeze | Unfreeze | Gap loss | lambda_gap | TTA | Trainable params | Best epoch | Val balanced metric finale |
|---:|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|---|---:|---:|---:|
| 18142 | L40S | 777 | 20 | 24 | 3e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.02 | True | 28,594,561 | 10 | 0.001061 |
| 18141 | L40S | 777 | 24 | 24 | 2e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.05 | True | 28,594,561 | 23 | 0.001063 |
| 18127 | L40S | 777 | 20 | 24 | 3e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.05 | True | 28,594,561 | 14 | 0.001063 |
| run002 | ancien run | non indiqué | 50 | 64 | 5e-6 | 1e-4 | cosine | 5e-7 | non indiqué | probablement 4* | non indiqué | non indiqué | non indiqué | 28,594,561 | non indiqué | 0.001616 |
| 18125 | H100 NVL | 123 | 20 | 16 | 8e-6 | 5e-5 | cosine | 1e-7 | False | 0 | True | 0.05 | True | 85,874,305 | 20 | 0.001138 |
| 18138 | L40S | 123 | 20 | 24 | 3e-5 | 5e-5 | cosine | 1e-7 | True | 4 | True | 0.05 | True | 28,594,561 | 6 | 0.001157 |

\* Pour `run002`, la carte fournie ne contient pas explicitement `freeze_backbone` ni `unfreeze_last_n_blocks`. Le nombre de paramètres entraînables `28,594,561` correspond cependant aux runs `u4` de DINOv3 ViT-B/16. Pour reproduire strictement l'ensemble final, l'artefact requis est le CSV `submissions/dinov3-vitb16_run002.csv`.

---

## Commandes de reproduction des runs Mesogip

À lancer depuis la racine du projet :

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

### Run002 historique

Fichier utilisé directement :

```text
submissions/dinov3-vitb16_run002.csv
```

Métadonnées disponibles :

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

## Commande exacte de construction de l'ensemble

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

## Vérification et rapatriement

```bash
ls -lh submissions/ensemble_final_bestguess_no_run001_run002_mesogip.csv
head -5 submissions/ensemble_final_bestguess_no_run001_run002_mesogip.csv
```

Depuis Git Bash local :

```bash
scp mesogip:~/datachallenge704/submissions/ensemble_final_bestguess_no_run001_run002_mesogip.csv .
```

---

## Notes de reproductibilité

- Tous les composants prédisent la colonne `FaceOcclusion`.
- Les fichiers sont alignés par colonne `ID` si elle existe ; sinon par ordre des lignes.
- La prédiction finale est clippée dans `[0, 1]`.
- Si une colonne `gender` existe dans le fichier de soumission, elle est fixée à `x` pour respecter le format de soumission.
- Le modèle final est un **ensemble post-hoc de CSV**, pas un checkpoint PyTorch unique.
