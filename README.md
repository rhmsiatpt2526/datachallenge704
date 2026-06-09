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
│   ├── datasets.py
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

Les images utilisées sont dans :

```text
crops/Crop_224_5fp_100K/
```

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

Pour l'instant, le backbone est gelé par défaut et seule la tête finale est entraînée.

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

## 6. Loss et métrique

La loss est une MSE pondérée :

```python
weights = 1 / 30 + y
loss = torch.mean(weights * (y_pred - y) ** 2)
```

La métrique de validation prend en compte les erreurs homme/femme :

```python
metric = (error_male + error_female) / 2 + abs(error_male - error_female)
```

Elle pénalise à la fois l'erreur globale et l'écart de performance entre les deux groupes.

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
* durée du run ;
* soumission ;
* checkpoint ;
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

La validation finale est toujours calculée à la fin du run.

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
logs/mobilenetv3_small/mobilenetv3_small_run006.md
```

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

Améliorations possibles pour la suite :

1. **Dégeler progressivement le backbone**
   Entraîner d'abord uniquement la tête, puis dégeler les derniers blocs du backbone.

2. **Ajouter de la data augmentation**
   Exemples : horizontal flip, color jitter, random crop léger, brightness/contrast augmentation.

3. **Comparer les architectures**
   MobileNetV3, EfficientNet, ConvNeXt, ResNet, DINOv2 et DINOv3.

4. **Sauvegarder le meilleur checkpoint**
   Sauvegarder le modèle qui obtient la meilleure métrique de validation, pas seulement le dernier modèle.

5. **Analyser les erreurs**
   Utiliser les prédictions de validation pour étudier les erreurs par genre, par niveau d'occlusion et par image.

6. **Ajouter un fichier de configuration**
   Utiliser un fichier YAML ou JSON pour enregistrer les paramètres de chaque expérience.

7. **Ensembling**
   Combiner plusieurs bons modèles pour réduire la variance des prédictions.

---

## 19. État actuel du projet

Le pipeline actuel permet de :

* charger les données ;
* entraîner plusieurs architectures pré-entraînées ;
* évaluer sur validation avec la métrique pondérée par genre ;
* utiliser un scheduler cosine avec `min_lr` ;
* contrôler la validation intermédiaire avec `val_every` ;
* générer une soumission ;
* sauvegarder un checkpoint ;
* logger automatiquement les résultats ;
* suivre les expériences avec MLflow ;
* exécuter l'entraînement sur cluster Slurm avec GPU CUDA ;
* tester DINOv2 dans l'environnement principal ;
* tester DINOv3 dans un environnement séparé dédié.

Ce projet constitue une base propre pour itérer sur les modèles et améliorer progressivement les performances.
