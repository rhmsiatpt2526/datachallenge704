# Data Challenge 704 — Face Occlusion Prediction

Projet réalisé dans le cadre du **Data Challenge 704**.

L'objectif est de prédire le niveau d'occlusion d'un visage à partir d'images prétraitées. La cible est une valeur continue :

```text
FaceOcclusion ∈ [0, 1]
```

Le problème est traité comme une tâche de **régression supervisée**. Le pipeline permet d'entraîner plusieurs modèles pré-entraînés, de générer une soumission, de sauvegarder des checkpoints, de logger les résultats dans des fichiers Markdown et de suivre les expériences avec **MLflow**.

---

## 1. Structure du projet

```text
datachallenge704/
│
├── checkpoints/                 # Checkpoints PyTorch, non versionnés
├── cluster_results/             # Résultats rapatriés depuis le cluster, non versionnés
├── crops/                       # Images cropées
│   └── Crop_224_5fp_100K/
├── logs/                        # Logs structurés des runs
├── mlartifacts/                 # Artefacts MLflow, non versionnés
├── mlflow.db                    # Base SQLite MLflow, non versionnée
├── notebooks/                   # Notebooks exploratoires
├── occlusion_datasets/          # CSV du challenge
│   ├── train.csv
│   └── test_students.csv
├── scripts/
│   ├── train_baseline.py        # Script principal d'entraînement
│   ├── job_script.sh            # Script Slurm standard
│   └── job_script_dinov3.sh     # Script Slurm dédié à l'environnement DINOv3
├── slurm_logs/                  # Logs Slurm archivés
├── src/
│   ├── config.py
│   ├── data.py
│   ├── dataset.py
│   ├── engine.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── predict.py
│   └── utils.py
├── submissions/                 # Fichiers de soumission
├── requirements.txt
├── README.md
├── task_brief.pdf
└── test_predictions.csv
```

À ne pas versionner :

```gitignore
mlflow.db
mlflow.db-shm
mlflow.db-wal
mlartifacts/
checkpoints/
cluster_results/
slurm-*.out
slurm-*.err
slurm-datachallenge704-*.out
slurm-datachallenge704-*.err
.venv/
.venv_dinov3/
```

---

## 2. Données

Les fichiers CSV sont dans :

```text
occlusion_datasets/train.csv
occlusion_datasets/test_students.csv
```

Quelques repères utiles :

* `train.csv` contient 100000 lignes ;
* `test_students.csv` contient 29980 lignes et une seule colonne `filename`.

Les images utilisées sont dans :

```text
crops/Crop_224_5fp_100K/
```

Les images sont des fichiers `.webp` au format 224x224 RGB.

Colonnes principales du `train.csv` :

* `filename` : nom de l'image ;
* `FaceOcclusion` : cible de régression ;
* `gender` : variable utilisée dans la métrique finale.

---

## 3. Modèles disponibles

Le modèle se choisit avec l'argument :

```bash
--model <nom_du_modele>
```

Modèles actuellement intégrés dans `src/model.py` :

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

La baseline historique est :

```text
mobilenetv3_small
```

Les modèles TorchVision utilisent les poids ImageNet. La tête finale est remplacée par une tête de régression finissant par un `Sigmoid`, afin de forcer les prédictions dans `[0, 1]`.

Le comportement du backbone est contrôlé par les arguments :

```bash
--freeze-backbone
--unfreeze-last-n-blocks <n>
```

Par défaut, `--freeze-backbone` n'est pas activé. Pour entraîner uniquement la tête de régression, il faut donc passer explicitement `--freeze-backbone`.

Pour faire du fine-tuning partiel, on peut combiner :

```bash
--freeze-backbone --unfreeze-last-n-blocks 2
```

Cela garde la majorité du backbone gelée, mais dégèle les derniers blocs compatibles du modèle. C'est particulièrement utile pour DINOv3 et les modèles de type ViT.

---

## 4. Environnements Python

Le projet utilise deux environnements séparés.

### Environnement principal : `.venv`

Utilisé pour les modèles stables :

```text
MobileNetV3
EfficientNet
ConvNeXt
ResNet
DINOv2
```

Création :

```bash
cd ~/datachallenge704
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pour DINOv2, l'environnement principal utilise une version stable de Transformers, par exemple :

```text
transformers==4.48.3
```

### Environnement expérimental : `.venv_dinov3`

Utilisé uniquement pour DINOv3. Il est séparé pour éviter de casser l'environnement stable.

Création recommandée sur le cluster :

```bash
cd ~/datachallenge704
python3.12 -m venv .venv_dinov3
source .venv_dinov3/bin/activate
python -m pip install --upgrade pip
```

Installation DINOv3 testée :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate pandas scikit-learn pillow tqdm mlflow
```

Versions validées pour charger DINOv3 :

```text
torch==2.12.0+cu126
transformers==5.10.2
```

Vérification :

```bash
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(hasattr(torch, 'float8_e8m0fnu'))"
```

Le résultat attendu pour DINOv3 est notamment :

```text
True
```

pour :

```python
hasattr(torch, 'float8_e8m0fnu')
```

---

## 5. DINOv2 et DINOv3

### DINOv2

DINOv2 est intégré via Hugging Face Transformers :

```text
facebook/dinov2-small
facebook/dinov2-base
```

Commande de test local très court :

```bash
python scripts/train_baseline.py \
    --model dinov2_base \
    --epochs 1 \
    --batch-size 2 \
    --num-workers 0 \
    --val-every 0 \
    --experiment-name test_dinov2_base
```

Sur CPU local, DINOv2 est très lent. Le vrai test doit être fait sur cluster GPU.

Commande cluster recommandée pour DINOv2 Base :

```bash
sbatch scripts/job_script.sh \
    --model dinov2_base \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name dinov2_base_e30
```

Si la mémoire GPU le permet, tester ensuite :

```bash
--batch-size 32
```

Run DINOv2 avec loss gender-gap :

```bash
sbatch scripts/job_script.sh \
    --model dinov2_base \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 5 \
    --use-gender-gap-loss \
    --lambda-gap 0.1 \
    --experiment-name dinov2_base_gender_gap_01
```

### DINOv3

DINOv3 est intégré de manière expérimentale via Hugging Face.

Modèles utilisés :

```text
facebook/dinov3-vits16-pretrain-lvd1689m
facebook/dinov3-vitb16-pretrain-lvd1689m
```

Les repos DINOv3 sont **gated** : il faut demander l'accès sur Hugging Face, puis se connecter sur le cluster.

Connexion Hugging Face :

```bash
cd ~/datachallenge704
source .venv_dinov3/bin/activate
hf auth login
```

Quand Hugging Face demande :

```text
Add token as git credential? [y/N]
```

répondre :

```text
N
```

Tester le chargement du modèle :

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python -c "from transformers import AutoModel; m=AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m'); print('ok')"
```

Si le test affiche :

```text
ok
```

l'accès Hugging Face et l'environnement DINOv3 sont fonctionnels.

---

## 6. Loss, métrique et sélection du meilleur checkpoint

### Loss principale

La loss de base est une MSE pondérée et normalisée par la somme des poids :

```python
weights = 1 / 30 + y
loss = torch.sum(weights * (y_pred - y) ** 2) / torch.sum(weights)
```

Cette pondération donne plus d'importance aux images avec une occlusion plus élevée.

### Loss optionnelle avec pénalité homme/femme

Le script permet aussi d'activer une loss plus alignée avec la métrique finale :

```bash
--use-gender-gap-loss --lambda-gap 0.1
```

La forme utilisée est :

```python
loss = weighted_mse + lambda_gap * abs(male_loss - female_loss)
```

avec :

```python
female_loss = weighted_mse sur les exemples female du batch
male_loss = weighted_mse sur les exemples male du batch
```

Si un batch ne contient qu'un seul genre, le script revient automatiquement à la MSE pondérée classique pour éviter une loss mal définie.

Valeurs raisonnables à tester :

```text
lambda_gap = 0.05
lambda_gap = 0.10
lambda_gap = 0.20
```

### Métrique de validation

La métrique de validation prend en compte les erreurs homme/femme :

```python
metric = (error_male + error_female) / 2 + abs(error_male - error_female)
```

Elle pénalise à la fois l'erreur moyenne et l'écart de performance entre les deux groupes.

### Best checkpoint

Pendant l'entraînement, le script sauvegarde automatiquement le meilleur checkpoint selon :

```text
validation_balanced_metric_epoch
```

Le fichier associé est :

```text
checkpoints/<model_name>/<run_tag>_best.pt
```

À la fin du run, ce meilleur checkpoint est rechargé avant de recalculer les métriques finales et de générer la soumission. La soumission finale correspond donc au meilleur modèle de validation, pas nécessairement au modèle de la dernière epoch.

---

## 7. Entraînement local

Test rapide :

```bash
python scripts/train_baseline.py \
    --model mobilenetv3_small \
    --epochs 1 \
    --batch-size 32 \
    --num-workers 0
```

Run local avec scheduler cosine :

```bash
python scripts/train_baseline.py \
    --model mobilenetv3_small \
    --epochs 10 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 0 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name mobilenetv3_small_baseline
```

Sur CPU local, garder généralement :

```bash
--num-workers 0
```

---

## 8. Entraînement sur cluster Slurm

Le script Slurm standard est :

```text
scripts/job_script.sh
```

Test court :

```bash
sbatch scripts/job_script.sh \
    --model mobilenetv3_small \
    --epochs 1 \
    --batch-size 64 \
    --num-workers 4 \
    --experiment-name test_cluster
```

Run complet MobileNetV3-Small :

```bash
sbatch scripts/job_script.sh \
    --model mobilenetv3_small \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name mobilenetv3_small_cosine_val10
```

Exemples de runs de comparaison :

```bash
sbatch scripts/job_script.sh \
    --model mobilenetv3_large \
    --epochs 30 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name night_mobilenetv3_large_e30
```

```bash
sbatch scripts/job_script.sh \
    --model efficientnet_b0 \
    --epochs 30 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name night_efficientnet_b0_e30
```

```bash
sbatch scripts/job_script.sh \
    --model efficientnet_b1 \
    --epochs 30 \
    --batch-size 96 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name night_efficientnet_b1_e30
```

```bash
sbatch scripts/job_script.sh \
    --model convnext_tiny \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name night_convnext_tiny_e30
```

Le script Slurm standard utilise généralement :

```bash
#SBATCH --partition=3090,P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=30:00:00
```

La partition `3090,P100` permet de soumettre le job sur l'une des deux partitions disponibles.

---

## 9. Entraînement DINOv3 sur cluster

DINOv3 utilise le script :

```text
scripts/job_script_dinov3.sh
```

Ce script doit utiliser :

```bash
VENV_DIR="${PROJECT_DIR}/.venv_dinov3"
PYTHON_BIN="${VENV_DIR}/bin/python"
```

et limiter les threads BLAS :

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Test Slurm DINOv3 court :

```bash
sbatch scripts/job_script_dinov3.sh \
    --model dinov3-vits16 \
    --epochs 1 \
    --batch-size 4 \
    --num-workers 4 \
    --val-every 0 \
    --experiment-name test_dinov3_vits16_venv
```

Run DINOv3 ViT-S/16 :

```bash
sbatch scripts/job_script_dinov3.sh \
    --model dinov3-vits16 \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name dinov3_vits16_e30
```

Run DINOv3 ViT-S/16 avec fine-tuning partiel des deux derniers blocs :

```bash
sbatch scripts/job_script_dinov3.sh \
    --model dinov3-vits16 \
    --epochs 30 \
    --batch-size 16 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 5 \
    --freeze-backbone \
    --unfreeze-last-n-blocks 2 \
    --experiment-name dinov3_vits16_unfreeze2_e30
```

Run DINOv3 ViT-S/16 avec fine-tuning partiel et loss gender-gap :

```bash
sbatch scripts/job_script_dinov3.sh \
    --model dinov3-vits16 \
    --epochs 30 \
    --batch-size 16 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 5 \
    --freeze-backbone \
    --unfreeze-last-n-blocks 2 \
    --use-gender-gap-loss \
    --lambda-gap 0.1 \
    --experiment-name dinov3_vits16_unfreeze2_gap01_e30
```

Run DINOv3 ViT-B/16, plus lourd :

```bash
sbatch scripts/job_script_dinov3.sh \
    --model dinov3-vitb16 \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name dinov3_vitb16_e30
```

Sur P100, commencer avec des batch sizes prudents.

---

## 10. Suivi Slurm

Voir ses jobs :

```bash
squeue -u $USER
```

Voir un job précis :

```bash
squeue -j <JOBID>
```

Historique d'un job :

```bash
sacct -j <JOBID>
```

Suivre la sortie standard :

```bash
tail -f slurm-datachallenge704-run001_<JOBID>.out
```

Suivre les warnings, erreurs ou sorties `tqdm` :

```bash
tail -f slurm-datachallenge704-run001_<JOBID>.err
```

Pour DINOv3 :

```bash
tail -f slurm-datachallenge704-dinov3_<JOBID>.out
tail -f slurm-datachallenge704-dinov3_<JOBID>.err
```

Annuler un job :

```bash
scancel <JOBID>
```

---

## 11. Suivi des expériences avec MLflow

Le script enregistre automatiquement dans MLflow :

* modèle utilisé ;
* hyperparamètres ;
* nombre de paramètres total et entraînable ;
* loss par epoch ;
* learning rate par epoch ;
* métriques train/validation finales ;
* métriques validation périodiques selon `--val-every` ;
* meilleure métrique de validation ;
* epoch du meilleur checkpoint ;
* durée du run ;
* soumission générée avec le meilleur modèle disponible ;
* checkpoint final ;
* meilleur checkpoint ;
* prédictions de validation, si activées dans le script ;
* log Markdown du run.

Le tracking utilise :

```text
mlflow.db
mlartifacts/
```

### Ouvrir MLflow en local

Depuis la racine du projet :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

### Lancer MLflow UI sur le cluster avec tunnel SSH

Terminal 1 : lancer le run Slurm.

Terminal 2, sur le cluster :

```bash
cd ~/datachallenge704
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p ~/tmp
TMPDIR=~/tmp mlflow ui \
    --backend-store-uri sqlite:///mlflow.db \
    --host 127.0.0.1 \
    --port 5001
```

Terminal 3, depuis l'ordinateur local :

```bash
ssh -L 5001:localhost:5001 cluster
```

Puis ouvrir :

```text
http://localhost:5001
```

Si le port est déjà utilisé :

```bash
ss -ltnp | grep -E '5000|5001'
kill <PID>
```

---

## 12. Paramètres principaux

Les principaux arguments du script sont :

```text
--model
--epochs
--batch-size
--lr
--weight-decay
--num-workers
--scheduler
--min-lr
--val-every
--experiment-name
--tracking-uri
--log-model
--freeze-backbone
--unfreeze-last-n-blocks
--use-gender-gap-loss
--lambda-gap
```

`--freeze-backbone` permet de geler le backbone et d'entraîner principalement la tête de régression.

`--unfreeze-last-n-blocks` permet de dégeler les derniers blocs du backbone lorsque le modèle le supporte. Exemple :

```bash
--freeze-backbone --unfreeze-last-n-blocks 2
```

`--use-gender-gap-loss` active une pénalité supplémentaire pour réduire l'écart d'erreur entre les genres.

`--lambda-gap` contrôle la force de cette pénalité. Exemple :

```bash
--use-gender-gap-loss --lambda-gap 0.1
```

### Scheduler

Le script accepte :

```text
--scheduler none/cosine
```

Avec `cosine`, le learning rate descend progressivement de `--lr` vers `--min-lr`.

Exemple :

```bash
--scheduler cosine --min-lr 1e-6
```

`min_lr=1e-6` est une valeur par défaut raisonnable. Elle évite que le learning rate tombe exactement à zéro en fin d'entraînement.

### Validation périodique

La validation intermédiaire est contrôlée par :

```text
--val-every
```

Exemples :

```bash
--val-every 10   # validation toutes les 10 epochs
--val-every 5    # validation toutes les 5 epochs
--val-every 0    # pas de validation intermédiaire
```

La validation finale est toujours calculée à la fin du run. Si un meilleur checkpoint a été sauvegardé pendant l'entraînement, il est rechargé avant le calcul final des métriques et avant la génération de la soumission.

Métriques périodiques visibles dans MLflow :

```text
validation_error_epoch
validation_female_error_epoch
validation_male_error_epoch
validation_gender_gap_epoch
validation_balanced_metric_epoch
```

---

## 13. Rapatrier les résultats depuis le cluster avec `scp`

Les commandes suivantes sont à lancer depuis **l'ordinateur local**, pas depuis le cluster.

Créer d'abord un dossier local pour stocker les résultats :

```bash
mkdir -p cluster_results
```

### Récupérer une soumission précise

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/submissions/mobilenetv3_small_run006.csv ./cluster_results/
```

### Récupérer le log du run correspondant

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/logs/mobilenetv3_small/mobilenetv3_small_run006.md ./cluster_results/
```

### Récupérer les logs Slurm du job correspondant

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_841668.out ./cluster_results/
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_841668.err ./cluster_results/
```

### Récupérer le checkpoint du modèle

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/checkpoints/mobilenetv3_small/mobilenetv3_small_run006.pt ./cluster_results/
```

### Récupérer MLflow

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlflow.db ./cluster_results/
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlartifacts ./cluster_results/
```

Ensuite, depuis `cluster_results` :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Récupérer tous les logs de runs, logs Slurm et MLflow

Depuis la racine du projet local :

```bash
cd ~/projets_ms_ia/datachallenge704 && \
mkdir -p cluster_results/slurm_logs cluster_results/run_logs && \
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_*.out ./cluster_results/slurm_logs/ && \
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_*.err ./cluster_results/slurm_logs/ && \
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-dinov3_*.out ./cluster_results/slurm_logs/ && \
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-dinov3_*.err ./cluster_results/slurm_logs/ && \
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/logs/* ./cluster_results/run_logs/ && \
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlflow.db ./cluster_results/ && \
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlartifacts ./cluster_results/
```

Sur Git Bash Windows, utiliser par exemple :

```bash
cd /c/Users/remi1/projets_ms_ia/datachallenge704
```

---

## 14. Exemple de résultat obtenu

Un run de 100 epochs sur P100 avec MobileNetV3-Small a produit :

```text
Validation balanced metric: 0.003603
```

Sorties associées :

```text
submissions/mobilenetv3_small_run006.csv
checkpoints/mobilenetv3_small/mobilenetv3_small_run006.pt
checkpoints/mobilenetv3_small/mobilenetv3_small_run006_best.pt
logs/mobilenetv3_small/mobilenetv3_small_run006.md
```

Selon les versions du script, le fichier `_best.pt` peut ne pas exister pour les anciens runs. Pour les nouveaux runs avec validation périodique, il est sauvegardé dès qu'une meilleure `validation_balanced_metric_epoch` est obtenue.

---

## 15. Reproductibilité

Les hyperparamètres principaux sont passés en ligne de commande et enregistrés dans :

* le fichier de log Markdown de chaque run ;
* MLflow.

Exemple local :

```bash
python scripts/train_baseline.py \
    --model mobilenetv3_small \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 0 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name mobilenetv3_small_baseline
```

Sur cluster :

```bash
sbatch scripts/job_script.sh \
    --model mobilenetv3_small \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name mobilenetv3_small_baseline
```

Exemple avec loss gender-gap :

```bash
sbatch scripts/job_script.sh \
    --model dinov2_base \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 5 \
    --use-gender-gap-loss \
    --lambda-gap 0.1 \
    --experiment-name dinov2_base_gender_gap_01
```

---

## 16. Commandes utiles

### Tester rapidement le pipeline

```bash
python scripts/train_baseline.py --epochs 1 --batch-size 32 --num-workers 0
```

### Lancer un run cluster

```bash
sbatch scripts/job_script.sh --epochs 100 --batch-size 128 --num-workers 4
```

### Ouvrir l'interface MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Voir les derniers logs de run

```bash
ls -lt logs/*/ | head
```

### Voir les dernières soumissions

```bash
ls -lt submissions/ | head
```

### Voir les derniers checkpoints

```bash
ls -lt checkpoints/*/ | head
```

### Lire le log d'un run

```bash
cat logs/mobilenetv3_small/mobilenetv3_small_run006.md
```

### Vérifier l'état du GPU

```bash
nvidia-smi
```

### Vérifier l'environnement DINOv3

```bash
source .venv_dinov3/bin/activate
python -c "import torch, transformers; print(torch.__version__); print(torch.cuda.is_available()); print(transformers.__version__); print(hasattr(torch, 'float8_e8m0fnu'))"
```

---

## 17. Remarques

Les barres de progression `tqdm` peuvent apparaître dans les fichiers `.err` Slurm. Ce n'est pas nécessairement une erreur.

Les erreurs OpenBLAS du type :

```text
OpenBLAS blas_thread_init: pthread_create failed
```

se corrigent généralement en limitant les threads :

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Un warning CuDNN peut également apparaître au premier entraînement :

```text
Applied workaround for CuDNN issue, install nvrtc.so
```

Tant que l'entraînement se termine correctement et que les fichiers de sortie sont générés, ce warning peut être ignoré dans un premier temps.

La base MLflow `mlflow.db`, le dossier `mlartifacts/`, les checkpoints et les environnements virtuels ne doivent pas être poussés sur GitHub.

---

## 18. Pistes d'amélioration

Améliorations déjà intégrées récemment :

1. **Fine-tuning contrôlé du backbone**
   Le script accepte `--freeze-backbone` et `--unfreeze-last-n-blocks`, ce qui permet de tester un entraînement head-only, un dégel partiel ou un entraînement plus complet.

2. **Loss optionnelle alignée avec la métrique finale**
   `--use-gender-gap-loss` ajoute une pénalité sur l'écart homme/femme via `--lambda-gap`.

3. **Sauvegarde du meilleur checkpoint**
   Le meilleur modèle selon `validation_balanced_metric_epoch` est sauvegardé dans un fichier `_best.pt`, puis rechargé avant la validation finale et la génération de la soumission.

4. **Suivi MLflow plus complet**
   Le nombre de paramètres total et entraînable, les métriques périodiques, les paramètres de fine-tuning et les paramètres de loss sont loggés.

Priorités restantes pour améliorer les performances :

1. **Comparer systématiquement les stratégies de fine-tuning**

   Tester pour DINOv3 ViT-S/16 :

   ```text
   head only
   unfreeze 2 blocs
   unfreeze 4 blocs
   ```

   avec des learning rates décroissants :

   ```text
   head only      : lr 1e-4
   unfreeze 2     : lr 5e-5
   unfreeze 4     : lr 2e-5
   ```

2. **Tester plusieurs valeurs de `lambda_gap`**

   ```text
   0.05
   0.10
   0.20
   ```

   Le choix doit se faire uniquement selon `validation_balanced_metric`, pas selon la loss train.

3. **Ajouter une augmentation légère de type RandomErasing**

   À tester prudemment, par exemple :

   ```python
   transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3))
   ```

   Il ne faut pas rendre cette augmentation trop forte, car la cible est justement un niveau d'occlusion.

4. **Analyser les erreurs**

   Utiliser les prédictions de validation pour étudier les erreurs par genre, par niveau d'occlusion et par image.

5. **Ensembling**

   Combiner les meilleurs modèles, par exemple :

   ```text
   DINOv3 ViT-S/16 unfreeze 2
   DINOv2 Base
   ConvNeXt Tiny
   ```

   puis moyenner les prédictions test.

6. **Ajouter un fichier de configuration**

   Utiliser un fichier YAML ou JSON pour rejouer facilement les meilleurs runs.

---

## 19. État actuel du projet

Le pipeline actuel permet de :

* charger les données ;
* entraîner plusieurs architectures pré-entraînées ;
* évaluer sur validation avec la métrique pondérée par genre ;
* utiliser un scheduler cosine avec `min_lr` ;
* contrôler la validation intermédiaire avec `val_every` ;
* utiliser une loss optionnelle avec pénalité homme/femme ;
* sauvegarder le meilleur checkpoint selon la métrique de validation ;
* recharger le meilleur checkpoint avant de générer la soumission ;
* générer une soumission ;
* sauvegarder un checkpoint final et un checkpoint best ;
* logger automatiquement les résultats ;
* suivre les expériences avec MLflow ;
* exécuter l'entraînement sur cluster Slurm avec GPU CUDA ;
* tester DINOv2 dans l'environnement principal ;
* tester DINOv3 dans un environnement séparé dédié.

Ce projet constitue une base propre pour itérer sur les modèles et améliorer progressivement les performances.
