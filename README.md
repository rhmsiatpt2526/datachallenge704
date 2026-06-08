# Data Challenge 704 — Face Occlusion Prediction

Projet réalisé dans le cadre du **Data Challenge 704**.

L'objectif est de prédire le niveau d'occlusion d'un visage à partir d'images prétraitées. La cible est une valeur continue :

```text
FaceOcclusion ∈ [0, 1]
```

Le problème est traité comme une tâche de **régression supervisée**.
Le pipeline permet d'entraîner plusieurs modèles pré-entraînés, de générer une soumission, de sauvegarder un checkpoint et de suivre les expériences avec **MLflow**.

---

## 1. Structure du projet

```text
datachallenge704/
│
├── checkpoints/                 # Checkpoints PyTorch
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
│   └── job_script.sh            # Script Slurm
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
slurm-*.out
slurm-*.err
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

```text
--model
```

Modèles actuellement intégrés ou prévus dans `src/model.py` :

```text
mobilenetv3_small
mobilenetv3_large
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

Les modèles TorchVision utilisent les poids ImageNet. La tête finale est remplacée par une tête de régression :

```python
nn.Sequential(
    nn.Linear(in_features, 1),
    nn.Sigmoid(),
)
```

Le `Sigmoid` force les prédictions dans `[0, 1]`.

Pour l'instant, le backbone est gelé et seule la tête finale est entraînée.

---

## 4. DINOv2 et DINOv3

### DINOv2

DINOv2 est intégré via Hugging Face Transformers :

```text
facebook/dinov2-small
facebook/dinov2-base
```

Pour éviter certains conflits entre versions récentes de `transformers` et PyTorch, le projet utilise actuellement :

```text
transformers==4.48.3
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
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name dinov2_base_frozen
```

Si la mémoire GPU le permet, tester ensuite :

```bash
--batch-size 32
```

### DINOv3

DINOv3 est intégré de manière expérimentale. Les modèles DINOv3 sont des repos Hugging Face gated : il faut demander l'accès sur Hugging Face et s'authentifier.

Commandes utiles :

```bash
hf auth login
```

ou :

```bash
huggingface-cli login
```

DINOv3 peut nécessiter une version plus récente de `transformers` que DINOv2. Pour l'instant, DINOv2 est donc le choix prioritaire et plus stable.

---

## 5. Loss et métrique

La loss est une MSE pondérée :

```python
weights = 1 / 30 + y
loss = sum(weights * (y_pred - y) ** 2) / sum(weights)
```

La métrique de validation prend en compte les erreurs homme/femme :

```python
metric = (error_male + error_female) / 2 + abs(error_male - error_female)
```

Elle pénalise à la fois l'erreur globale et l'écart de performance entre les deux groupes.

---

## 6. Installation

Depuis la racine du projet :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows avec Git Bash :

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

Vérifier CUDA :

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Vérifier MLflow :

```bash
mlflow --version
```

Vérifier Hugging Face :

```bash
hf auth whoami
```

---

## 7. Entraînement local

Test rapide avec MobileNetV3-Small :

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

Le script Slurm est :

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

Run complet recommandé :

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

Exemple avec ConvNeXt-Tiny :

```bash
sbatch scripts/job_script.sh \
    --model convnext_tiny \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name convnext_tiny_cosine_val10
```

Exemple avec DINOv2 Base :

```bash
sbatch scripts/job_script.sh \
    --model dinov2_base \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name dinov2_base_frozen
```

Le script Slurm utilise actuellement :

```bash
#SBATCH --partition=3090,P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=30:00:00
```

La partition `3090,P100` permet de soumettre le job sur l'une des deux partitions disponibles.

---

## 9. Suivi Slurm

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

Suivre les warnings / erreurs / barres `tqdm` :

```bash
tail -f slurm-datachallenge704-run001_<JOBID>.err
```

Annuler un job :

```bash
scancel <JOBID>
```

---

## 10. Suivi des expériences avec MLflow

Le script enregistre automatiquement dans MLflow :

* modèle utilisé ;
* hyperparamètres ;
* loss par epoch ;
* learning rate par epoch ;
* métriques train/validation finales ;
* métriques validation périodiques selon `--val-every` ;
* durée du run ;
* soumission ;
* checkpoint ;
* log `.md`.

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

### Voir MLflow en direct depuis le cluster

Utiliser trois terminaux.

Terminal 1 — lancer le run :

```bash
ssh cluster
cd ~/datachallenge704

sbatch scripts/job_script.sh \
    --model dinov2_base \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 4 \
    --scheduler cosine \
    --val-every 10 \
    --experiment-name dinov2_base_frozen
```

Terminal 2 — lancer MLflow UI sur le cluster :

```bash
ssh cluster
cd ~/datachallenge704
source .venv/bin/activate
```

Limiter les threads pour éviter les erreurs OpenBLAS :

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Lancer MLflow sur le port `5001` :

```bash
mkdir -p ~/tmp

TMPDIR=~/tmp mlflow ui \
    --backend-store-uri sqlite:///mlflow.db \
    --host 127.0.0.1 \
    --port 5001
```

Terminal 3 — créer le tunnel SSH :

```bash
ssh -L 5001:localhost:5001 cluster
```

Ensuite ouvrir localement :

```text
http://localhost:5001
```

Si `~/.ssh/config` contient déjà :

```text
LocalForward 5001 localhost:5001
```

alors un simple :

```bash
ssh cluster
```

suffit.

---

## 11. Scheduler et validation périodique

Le script accepte :

```text
--scheduler none/cosine
```

Avec `cosine`, le learning rate descend progressivement de `--lr` vers `--min-lr`.

Exemple :

```bash
--scheduler cosine --min-lr 1e-6
```

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

## 12. Fichiers générés après un run

### Soumission

```text
submissions/<model_name>_runXXX.csv
```

Exemple :

```text
submissions/mobilenetv3_small_run006.csv
```

### Checkpoint

```text
checkpoints/<model_name>/<model_name>_runXXX.pt
```

Exemple :

```text
checkpoints/mobilenetv3_small/mobilenetv3_small_run006.pt
```

### Log de run

```text
logs/<model_name>/<model_name>_runXXX.md
```

Exemple :

```text
logs/mobilenetv3_small/mobilenetv3_small_run006.md
```

### MLflow

```text
mlflow.db
mlartifacts/
```

### Logs Slurm

```text
slurm-datachallenge704-run001_<JOBID>.out
slurm-datachallenge704-run001_<JOBID>.err
```

---

## 13. Rapatrier les résultats depuis le cluster avec `scp`

Les commandes suivantes sont à lancer depuis **l'ordinateur local**.

Créer un dossier de résultats :

```bash
mkdir cluster_results
```

Récupérer une soumission :

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/submissions/mobilenetv3_small_run006.csv ./cluster_results/
```

Récupérer le log du run :

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/logs/mobilenetv3_small/mobilenetv3_small_run006.md ./cluster_results/
```

Récupérer les logs Slurm :

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_<JOBID>.out ./cluster_results/
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_<JOBID>.err ./cluster_results/
```

Récupérer le checkpoint :

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/checkpoints/mobilenetv3_small/mobilenetv3_small_run006.pt ./cluster_results/
```

Récupérer MLflow :

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlflow.db ./cluster_results/
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlartifacts ./cluster_results/
```

Puis ouvrir MLflow localement :

```bash
cd cluster_results
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Navigateur :

```text
http://127.0.0.1:5000
```

Récupérer tous les fichiers utiles :

```bash
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/submissions ./cluster_results/
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/logs ./cluster_results/
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/checkpoints ./cluster_results/
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlartifacts ./cluster_results/
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlflow.db ./cluster_results/
```

---

## 14. Reproductibilité

Les arguments principaux sont enregistrés dans :

* le log `.md` ;
* le checkpoint ;
* MLflow.

Arguments principaux :

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

Exemple complet :

```bash
sbatch scripts/job_script.sh \
    --model dinov2_base \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --val-every 10 \
    --experiment-name dinov2_base_frozen
```

---

## 15. Commandes utiles

Tester rapidement :

```bash
python scripts/train_baseline.py \
    --model mobilenetv3_small \
    --epochs 1 \
    --batch-size 32 \
    --num-workers 0
```

Tester DINOv2 Base localement :

```bash
python scripts/train_baseline.py \
    --model dinov2_base \
    --epochs 1 \
    --batch-size 2 \
    --num-workers 0 \
    --val-every 0 \
    --experiment-name test_dinov2_base
```

Lancer un run cluster DINOv2 Base :

```bash
sbatch scripts/job_script.sh \
    --model dinov2_base \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 4 \
    --scheduler cosine \
    --val-every 10 \
    --experiment-name dinov2_base_frozen
```

Voir les dernières soumissions :

```bash
ls -lt submissions/ | head
```

Voir les derniers logs :

```bash
ls -lt logs/ | head
```

Voir les derniers checkpoints :

```bash
ls -lt checkpoints/ | head
```

Vérifier le GPU :

```bash
nvidia-smi
```

---

## 16. Remarques

Les barres `tqdm` peuvent apparaître dans les fichiers `.err` Slurm. Ce n'est pas forcément une erreur.

Un warning CuDNN peut apparaître :

```text
Applied workaround for CuDNN issue, install nvrtc.so
```

Tant que l'entraînement se termine correctement, ce warning peut être ignoré.

Sur Windows, Hugging Face peut afficher un warning lié aux symlinks. Il n'est pas bloquant.

---

## 17. Pistes d'amélioration

Pistes envisagées :

* sauvegarder le meilleur checkpoint validation ;
* sauvegarder les prédictions de validation ;
* créer un notebook d'analyse d'erreurs ;
* dégeler progressivement le backbone ;
* ajouter de la data augmentation ;
* tester MobileNetV3-Large ;
* tester EfficientNet-B1 ;
* tester ConvNeXt-Tiny ;
* tester DINOv2 Base ;
* tester DINOv3 dès que l'accès Hugging Face est autorisé ;
* faire de la cross-validation ;
* ajouter un fichier de configuration YAML.
