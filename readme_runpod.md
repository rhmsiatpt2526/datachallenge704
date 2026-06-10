# Guide RunPod pour entraîner DINOv3

Ce guide décrit une installation simple et robuste pour lancer les runs DINOv3 du projet `datachallenge704` sur RunPod, en particulier avec une H100.

## 1. Configuration conseillée du pod

Dans RunPod, choisir :

- GPU : **H100 SXM 80GB** si disponible.
- Template : **RunPod PyTorch 2.4.0** ou image PyTorch CUDA récente déjà fonctionnelle.
- GPU count : **1**.
- Volume disk : **100 Go minimum**.
- Start Jupyter notebook : activé.
- SSH access : recommandé.

Éviter de multiplier les GPU : le code actuel n’est pas configuré pour du multi-GPU.

## 2. Cloner le dépôt

Dans le terminal RunPod :

```bash
cd /workspace
git clone https://github.com/rhmsiatpt2526/datachallenge704.git
cd datachallenge704
```

Si le dépôt existe déjà :

```bash
cd /workspace/datachallenge704
git status
git pull
```

Attention : ne pas faire `git pull` si des fichiers ont été modifiés directement sur RunPod sans commit, sinon risque de conflit ou d’écrasement.

## 3. Ne pas créer de virtualenv

Sur l’image PyTorch de RunPod, PyTorch + CUDA sont déjà installés correctement.

Ne pas faire :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_dinov3.txt
```

Et surtout ne pas installer directement :

```bash
pip install -r requirements_dinov3.txt
```

Ce fichier peut contenir des dépendances spécifiques au cluster, par exemple :

```text
torch==2.12.0+cu126
cuda-toolkit==12.6.3
triton==3.7.0
```

Ces versions peuvent casser l’environnement RunPod.

## 4. Vérifier que PyTorch voit bien la H100

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

Sortie attendue :

```text
cuda: True
gpu: NVIDIA H100 80GB HBM3
```

## 5. Installer les dépendances utiles

Installer uniquement les librairies nécessaires sans toucher manuellement à Torch/CUDA :

```bash
pip install --upgrade pip
pip install --ignore-installed mlflow scikit-learn pandas pillow tqdm opencv-python matplotlib seaborn
```

Pour DINOv3, utiliser une version stable de `transformers` compatible avec l’environnement testé :

```bash
pip uninstall -y transformers huggingface_hub tokenizers
pip install "transformers==4.56.2" "huggingface_hub==0.35.3" "tokenizers==0.22.0"
```

Vérifier :

```bash
python - <<'PY'
import torch
import transformers
import huggingface_hub
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 6. Se connecter à Hugging Face

DINOv3 est un modèle gated. Il faut être connecté à Hugging Face avec un compte ayant accepté l’accès au modèle.

Vérifier :

```bash
hf auth whoami
```

Si non connecté :

```bash
hf auth login
```

Coller le token Hugging Face, puis vérifier :

```bash
hf auth whoami
```

Erreur typique si non connecté ou accès non accepté :

```text
401 Unauthorized
```

## 7. Vérifier que le code compile

```bash
cd /workspace/datachallenge704
python -m py_compile src/dataset.py src/data.py src/model.py src/predict.py scripts/train_baseline.py
```

Si cette commande ne renvoie rien, c’est bon.

## 8. Télécharger et vérifier le dataset

Depuis la racine du projet :

```bash
cd /workspace/datachallenge704
wget -O DataChallengeTelecom.zip "https://partage.imt.fr/index.php/s/ntYk27ZFCbeKGqW/download"
unzip DataChallengeTelecom.zip
```

Le projet actuel attend typiquement ces chemins dans `src/config.py` :

```text
/workspace/datachallenge704/occlusion_datasets/train.csv
/workspace/datachallenge704/occlusion_datasets/test_students.csv
/workspace/datachallenge704/crops/Crop_224_5fp_100K
```

Vérifier la structure :

```bash
find . -maxdepth 3 -type f | head -50
find . -iname "*.csv"
find . -type d | grep -i crop
cat src/config.py
```

Vérifier que les fichiers attendus existent :

```bash
ls -lh occlusion_datasets/train.csv
ls -lh occlusion_datasets/test_students.csv
ls -ld crops/Crop_224_5fp_100K
```

Vérifier le nombre d’images :

```bash
find crops/Crop_224_5fp_100K -type f | wc -l
du -sh crops
```

Valeur déjà observée sur RunPod :

```text
129980 images
```

Un écart de taille disque entre local et RunPod peut être normal selon le système de fichiers. Le plus important est qu’aucune image référencée par les CSV ne manque.

Validation complète :

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

train_csv = Path("/workspace/datachallenge704/occlusion_datasets/train.csv")
test_csv = Path("/workspace/datachallenge704/occlusion_datasets/test_students.csv")
image_dir = Path("/workspace/datachallenge704/crops/Crop_224_5fp_100K")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

missing_train = [f for f in df_train["filename"] if not (image_dir / f).exists()]
missing_test = [f for f in df_test["filename"] if not (image_dir / f).exists()]

print("train rows:", len(df_train))
print("test rows:", len(df_test))
print("missing train images:", len(missing_train))
print("missing test images:", len(missing_test))
PY
```

Sortie attendue :

```text
train rows: 100000
test rows: 29980
missing train images: 0
missing test images: 0
```

## 9. Vérifier le bug train/validation

La validation doit utiliser les labels mais pas les augmentations.

Configuration correcte :

```text
train      : training=True,  augment=True
validation : training=True,  augment=False
test       : training=False, augment=False
```

Si le code plante avec :

```text
ValueError: not enough values to unpack (expected 4, got 2)
```

cela signifie probablement que le validation set a été créé avec `training=False`, donc il ne renvoie pas `x, y, gender, filename`.

Dans `src/data.py`, la validation doit ressembler à :

```python
validation_set = OcclusionDataset(
    df_val,
    IMAGE_DIR,
    training=True,
    augment=False,
)
```

## 10. Vérifier le bug du best checkpoint

Si le run plante avec :

```text
KeyError: 'error'
```

dans `save_checkpoint`, c’est que le best checkpoint est sauvegardé avec `train_stats={}` alors que `save_checkpoint` attend `train_stats["error"]`.

Le bloc de sauvegarde du best checkpoint doit utiliser un `torch.save(...)` direct, par exemple :

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_tag": run_tag,
        "model_name": MODEL_NAME,
        "args": vars(args),
        "epoch": epoch + 1,
        "best_val_metric": best_val_metric,
        "best_val_stats": val_stats_epoch,
    },
    best_checkpoint_path,
)
```

Après modification :

```bash
python -m py_compile scripts/train_baseline.py
```

## 11. Lancer un smoke test

Avant de lancer un run long, faire un test d’une epoch :

```bash
python scripts/train_baseline.py \
  --model dinov3-vitb16 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 2 \
  --epochs 1 \
  --lr 1e-5 \
  --batch-size 64 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --val-every 1 \
  --experiment-name runpod_smoke_test
```

Ce smoke test doit aller jusqu’au bout :

```text
Submission saved to: ...
Checkpoint saved to: ...
Run log saved to: ...
Best epoch: ...
Validation balanced metric: ...
```

Si CUDA out of memory :

```bash
--batch-size 32
```

## 12. Paramètres de vitesse recommandés

Sur H100, les tests ont montré :

```text
batch 64,  workers 8  : stable, bon choix par défaut
batch 128, workers 16 : fonctionne, mais gain faible et loss moins favorable sur smoke test
```

Choix safe recommandé :

```bash
--batch-size 64 --num-workers 8
```

On peut tester `--num-workers 16`, mais si les performances ne s’améliorent pas, revenir à `8`.

## 13. Installer et utiliser tmux

`tmux` peut ne pas être installé par défaut.

Installer :

```bash
apt update
apt install -y tmux
```

Créer une session :

```bash
cd /workspace/datachallenge704
tmux new -s train
```

Lancer le run dans la session.

Se détacher sans arrêter :

```text
Ctrl+B puis D
```

Lister les sessions :

```bash
tmux ls
```

Revenir dans la session :

```bash
tmux attach -t train
```

Arrêter un run en cours depuis la session :

```text
Ctrl+C
```

Alternative sans `tmux` :

```bash
cd /workspace/datachallenge704

nohup python scripts/train_baseline.py \
  --model dinov3-vitb16 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 4 \
  --epochs 50 \
  --lr 5e-6 \
  --batch-size 64 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 5e-7 \
  --val-every 5 \
  --experiment-name h100_dinov3_vitb16_degel4_bs64_aug \
  > run_h100_degel4.log 2>&1 &
```

Suivre le log :

```bash
tail -f run_h100_degel4.log
```

## 14. Run principal recommandé

Run fort, avec 4 derniers blocs DINOv3 dégelés :

```bash
python scripts/train_baseline.py \
  --model dinov3-vitb16 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 4 \
  --epochs 50 \
  --lr 5e-6 \
  --batch-size 64 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 5e-7 \
  --val-every 5 \
  --experiment-name h100_dinov3_vitb16_degel4_bs64_aug
```

Run plus stable, avec seulement 2 derniers blocs dégelés :

```bash
python scripts/train_baseline.py \
  --model dinov3-vitb16 \
  --freeze-backbone \
  --unfreeze-last-n-blocks 2 \
  --epochs 50 \
  --lr 1e-5 \
  --batch-size 64 \
  --num-workers 8 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --val-every 5 \
  --experiment-name h100_dinov3_vitb16_degel2_bs64_aug
```

## 15. Résultat de référence observé

Run H100 `dinov3-vitb16`, 4 derniers blocs dégelés, 50 epochs :

```text
Best epoch: 45
Validation balanced metric: 0.001616
Submission: submissions/dinov3-vitb16_run002.csv
Best checkpoint: checkpoints/dinov3-vitb16/dinov3-vitb16_run002_best.pt
Final checkpoint: checkpoints/dinov3-vitb16/dinov3-vitb16_run002.pt
Log: logs/dinov3-vitb16/dinov3-vitb16_run002.md
```

Le checkpoint `_best.pt` est celui à privilégier pour le meilleur modèle de validation. Le CSV de soumission généré après rechargement du best checkpoint est normalement le bon fichier à soumettre.

## 16. Surveiller les résultats

Les fichiers importants sont générés dans :

```text
submissions/
checkpoints/
logs/
mlflow.db
mlartifacts/
```

Afficher les derniers logs :

```bash
ls -lh submissions
ls -lh checkpoints/dinov3-vitb16
ls -lh logs/dinov3-vitb16
```

Lire le log markdown du run :

```bash
cat logs/dinov3-vitb16/dinov3-vitb16_run002.md
```

## 17. Récupérer les résultats sur le PC local

### Option A : récupérer seulement un run

Depuis le PC local, adapter `IP`, `PORT` et le chemin local :

```bash
mkdir -p ~/projets_ms_ia/datachallenge704/runpod_results/run002 && \
scp -P PORT -i ~/.ssh/id_ed25519 \
  root@IP:/workspace/datachallenge704/submissions/dinov3-vitb16_run002.csv \
  root@IP:/workspace/datachallenge704/logs/dinov3-vitb16/dinov3-vitb16_run002.md \
  root@IP:/workspace/datachallenge704/checkpoints/dinov3-vitb16/dinov3-vitb16_run002.pt \
  root@IP:/workspace/datachallenge704/checkpoints/dinov3-vitb16/dinov3-vitb16_run002_best.pt \
  ~/projets_ms_ia/datachallenge704/runpod_results/run002/
```

Exemple :

```bash
scp -P 16191 -i ~/.ssh/id_ed25519 \
  root@64.247.201.48:/workspace/datachallenge704/submissions/dinov3-vitb16_run002.csv \
  root@64.247.201.48:/workspace/datachallenge704/logs/dinov3-vitb16/dinov3-vitb16_run002.md \
  root@64.247.201.48:/workspace/datachallenge704/checkpoints/dinov3-vitb16/dinov3-vitb16_run002.pt \
  root@64.247.201.48:/workspace/datachallenge704/checkpoints/dinov3-vitb16/dinov3-vitb16_run002_best.pt \
  ~/projets_ms_ia/datachallenge704/runpod_results/run002/
```

### Option B : créer une archive sur RunPod

Sur RunPod :

```bash
cd /workspace/datachallenge704
tar -czf runpod_run002_results.tar.gz \
  submissions/dinov3-vitb16_run002.csv \
  logs/dinov3-vitb16/dinov3-vitb16_run002.md \
  checkpoints/dinov3-vitb16/dinov3-vitb16_run002.pt \
  checkpoints/dinov3-vitb16/dinov3-vitb16_run002_best.pt \
  mlflow.db \
  mlartifacts
```

Puis depuis le PC local :

```bash
scp -P PORT -i ~/.ssh/id_ed25519 \
  root@IP:/workspace/datachallenge704/runpod_run002_results.tar.gz \
  ~/projets_ms_ia/datachallenge704/
```

Décompresser localement :

```bash
cd ~/projets_ms_ia/datachallenge704
tar -xzf runpod_run002_results.tar.gz
```

### Option C : archiver tout

Sur RunPod :

```bash
cd /workspace/datachallenge704
tar -czf results_runpod_all.tar.gz submissions checkpoints logs mlflow.db mlartifacts
```

Puis depuis le PC local :

```bash
scp -P PORT -i ~/.ssh/id_ed25519 \
  root@IP:/workspace/datachallenge704/results_runpod_all.tar.gz \
  ~/projets_ms_ia/datachallenge704/
```

## 18. Arrêter la facturation

Quand les résultats sont récupérés :

1. Stopper ou terminer le pod dans RunPod.
2. Supprimer les volumes inutiles si le challenge est terminé.
3. Vérifier que le pod n’est plus en cours d’exécution.
4. Vérifier dans Billing que les paiements automatiques ne risquent pas de relancer une consommation non souhaitée.

À retenir : sur RunPod, ne pas utiliser `requirements_dinov3.txt` tel quel. Il est adapté au cluster, pas à l’image PyTorch RunPod.
