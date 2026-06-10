# Guide RunPod pour entraîner DINOv3

Ce guide décrit une installation simple et robuste pour lancer les runs DINOv3 du projet `datachallenge704` sur RunPod, en particulier avec une H100.

## 1. Configuration conseillée du pod

Dans RunPod, choisir :

- GPU : **H100 SXM 80GB** si disponible.
- Template : **RunPod PyTorch 2.4.0**.
- GPU count : **1**.
- Volume disk : **100 Go minimum**.
- Start Jupyter notebook : activé.
- SSH access : recommandé, mais pas obligatoire.

Éviter de multiplier les GPU : le code actuel n’est pas configuré pour du multi-GPU.

## 2. Cloner le dépôt

Dans le terminal RunPod :

```bash
cd /workspace
git clone https://github.com/rhmsiatpt2526/datachallenge704.git
cd datachallenge704
```

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

## 5. Installer uniquement les dépendances utiles

Installer les librairies nécessaires sans toucher manuellement à Torch/CUDA :

```bash
pip install --upgrade pip
pip install --ignore-installed transformers mlflow scikit-learn pandas pillow tqdm opencv-python matplotlib seaborn
```

L’option `--ignore-installed` évite certains conflits avec les paquets système déjà présents dans l’image RunPod.

## 6. Vérifier que le code compile

```bash
python -m py_compile src/dataset.py src/data.py src/model.py scripts/train_baseline.py
```

Si cette commande ne renvoie rien, c’est bon.

## 7. Télécharger le dataset

Depuis la racine du projet :

```bash
cd /workspace/datachallenge704
wget -O DataChallengeTelecom.zip "https://partage.imt.fr/index.php/s/ntYk27ZFCbeKGqW/download"
unzip DataChallengeTelecom.zip
```

Inspecter ensuite ce qui a été extrait :

```bash
find . -maxdepth 3 -type f | head -50
find . -iname "*.csv"
find . -type d | grep -i image
cat src/config.py
```

Le but est de placer les fichiers aux chemins attendus par `src/config.py`.

Exemple si l’archive contient un dossier `occlusion_datasets` :

```bash
mkdir -p data
cp occlusion_datasets/train.csv data/
cp occlusion_datasets/test.csv data/
cp -r occlusion_datasets/images data/
```

Adapter les noms si l’archive a une structure différente.

Vérification :

```bash
ls data
find data -maxdepth 2 -type f | head
du -sh data
```

## 8. Lancer un smoke test

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

Si erreur CUDA out of memory, relancer avec :

```bash
--batch-size 32
```

## 9. Run principal recommandé

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
  --experiment-name h100_dinov3_vitb16_degel4_aug
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
  --experiment-name h100_dinov3_vitb16_degel2_aug
```

## 10. Surveiller les résultats

Les fichiers importants sont générés dans :

```text
submissions/
checkpoints/
logs/
mlflow.db
mlartifacts/
```

Le checkpoint `_best.pt` est celui à privilégier si la sauvegarde du meilleur modèle est activée dans `train_baseline.py`.

## 11. Archiver les résultats avant d’arrêter le pod

À la fin des runs :

```bash
tar -czf results_runpod.tar.gz submissions checkpoints logs mlflow.db mlartifacts
```

Télécharger ensuite `results_runpod.tar.gz` avant de supprimer le pod.

## 12. Arrêter la facturation

Quand les résultats sont récupérés :

1. Stopper ou terminer le pod dans RunPod.
2. Supprimer les volumes inutiles si le challenge est terminé.
3. Vérifier que le pod n’est plus en cours d’exécution.

À retenir : sur RunPod, ne pas utiliser `requirements_dinov3.txt` tel quel. Il est adapté au cluster, pas à l’image PyTorch RunPod.
