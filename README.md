# Data Challenge 704 — Face Occlusion Prediction

Ce projet a été réalisé dans le cadre du **Data Challenge 704**.

L'objectif est de prédire le niveau d'occlusion d'un visage à partir d'images prétraitées. La cible à prédire est une valeur continue :

```text
FaceOcclusion ∈ [0, 1]
```

Le problème est traité comme une tâche de **régression supervisée**.
Le pipeline actuel entraîne une baseline basée sur **MobileNetV3-Small pré-entraîné sur ImageNet**, puis génère automatiquement :

* une soumission `.csv` ;
* un checkpoint PyTorch `.pt` ;
* un fichier de log `.md` contenant les métriques du run ;
* des logs Slurm `.out` et `.err` lors de l'exécution sur cluster ;
* un suivi d'expérience avec **MLflow**.

---

## 1. Structure du projet

```text
datachallenge704/
│
├── checkpoints/                 # Checkpoints PyTorch générés après entraînement
│   └── mobilenetv3_small/
│
├── crops/                       # Images cropées utilisées pour l'entraînement
│   └── Crop_224_5fp_100K/
│
├── logs/                        # Logs structurés des runs
│   └── mobilenetv3_small/
│
├── mlartifacts/                 # Artefacts MLflow, non versionnés
├── mlflow.db                    # Base SQLite MLflow, non versionnée
│
├── notebooks/                   # Notebooks exploratoires
│
├── occlusion_datasets/          # Fichiers CSV du challenge
│   ├── train.csv
│   └── test_students.csv
│
├── scripts/
│   ├── train_baseline.py        # Point d'entrée principal pour l'entraînement
│   └── job_script.sh            # Script Slurm pour le cluster
│
├── slurm_logs/                  # Logs Slurm rapatriés ou archivés
│
├── src/
│   ├── config.py                # Chemins et constantes globales
│   ├── data.py                  # Chargement des données et DataLoaders
│   ├── datasets.py              # Dataset PyTorch
│   ├── engine.py                # Boucle d'entraînement
│   ├── losses.py                # Fonction de loss
│   ├── metrics.py               # Métriques de validation
│   ├── model.py                 # Construction du modèle
│   ├── predict.py               # Fonctions de prédiction
│   └── utils.py                 # Fonctions utilitaires : logs, checkpoints, run id
│
├── submissions/                 # Fichiers CSV de soumission
│
├── requirements.txt             # Dépendances Python
├── README.md
├── task_brief.pdf               # Sujet du challenge
└── test_predictions.csv         # Ancien fichier de prédiction / référence de travail
```

Les fichiers MLflow suivants ne doivent pas être versionnés :

```text
mlflow.db
mlflow.db-shm
mlflow.db-wal
mlartifacts/
```

Ils doivent être ajoutés au `.gitignore`.

---

## 2. Données utilisées

Les fichiers tabulaires sont stockés dans :

```text
occlusion_datasets/train.csv
occlusion_datasets/test_students.csv
```

Le dossier d'images utilisé par le pipeline est :

```text
crops/Crop_224_5fp_100K/
```

Le fichier `train.csv` contient notamment :

* `filename` : nom du fichier image ;
* `FaceOcclusion` : cible de régression ;
* `gender` : variable utilisée dans la métrique finale.

Le fichier `test_students.csv` contient les images pour lesquelles une prédiction doit être générée.

---

## 3. Modèle utilisé

La baseline actuelle repose sur :

```text
MobileNetV3-Small
```

Le modèle est chargé avec les poids pré-entraînés ImageNet :

```python
MobileNet_V3_Small_Weights.DEFAULT
```

La tête de classification d'origine est remplacée par une tête de régression :

```python
nn.Sequential(
    nn.Linear(in_features, 1),
    nn.Sigmoid(),
)
```

Le `Sigmoid` force les prédictions à rester dans l'intervalle `[0, 1]`, ce qui correspond au domaine de la cible `FaceOcclusion`.

Pour la baseline actuelle, le backbone est gelé :

```python
for param in model.features.parameters():
    param.requires_grad = False
```

Seule la tête finale est entraînée.

---

## 4. Loss et métrique

La loss utilisée à l'entraînement est une MSE pondérée :

```python
weights = 1 / 30 + y
loss = sum(weights * (y_pred - y) ** 2) / sum(weights)
```

Cette loss est alignée avec la fonction d'erreur utilisée pour l'évaluation.

La métrique finale prend en compte les erreurs séparées sur les groupes homme/femme :

```python
metric = (error_male + error_female) / 2 + abs(error_male - error_female)
```

Cette formulation pénalise à la fois :

* l'erreur globale ;
* l'écart de performance entre les deux groupes.

---

## 5. Installation

Depuis la racine du projet :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sur Windows avec Git Bash :

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

Pour vérifier que PyTorch voit bien le GPU CUDA :

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Pour vérifier que MLflow est installé :

```bash
mlflow --version
```

---

## 6. Lancer un entraînement en local

Pour tester rapidement le pipeline en local :

```bash
python scripts/train_baseline.py --epochs 1 --batch-size 32 --num-workers 0
```

Pour un entraînement plus long en local :

```bash
python scripts/train_baseline.py \
    --epochs 10 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 0
```

Sur une machine locale CPU, il est recommandé de garder :

```bash
--num-workers 0
```

---

## 7. Suivi des expériences avec MLflow

Le projet utilise **MLflow** pour suivre proprement les runs d'entraînement.

À chaque run, le script enregistre automatiquement :

* les hyperparamètres ;
* la loss d'entraînement par epoch ;
* les métriques finales train/validation ;
* la durée du run ;
* la soumission `.csv` ;
* le checkpoint `.pt` ;
* le log `.md` du run.

Le tracking MLflow utilise actuellement une base SQLite locale :

```text
mlflow.db
```

et un dossier d'artefacts :

```text
mlartifacts/
```

Ces fichiers sont générés automatiquement au premier run MLflow.

### Lancer un run avec MLflow en local

```bash
python scripts/train_baseline.py \
    --epochs 1 \
    --batch-size 32 \
    --num-workers 0 \
    --experiment-name test_mlflow
```

Pour un run plus long :

```bash
python scripts/train_baseline.py \
    --epochs 10 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 0 \
    --experiment-name mobilenetv3_small_baseline
```

### Lancer un run MLflow sur le cluster

```bash
sbatch scripts/job_script.sh \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --experiment-name mobilenetv3_small_baseline
```

Les arguments passés à `job_script.sh` sont transmis directement à `scripts/train_baseline.py`.

### Ouvrir l'interface MLflow en local

Depuis la racine du projet :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Puis ouvrir dans un navigateur :

```text
http://127.0.0.1:5000
```

L'interface permet de comparer les runs selon :

* `epochs` ;
* `batch_size` ;
* `lr` ;
* `weight_decay` ;
* `train_epoch_loss` ;
* `validation_error` ;
* `validation_balanced_metric` ;
* `validation_gender_gap` ;
* `run_duration_seconds`.

### Logger le modèle complet dans MLflow

Par défaut, le script sauvegarde déjà un checkpoint PyTorch dans :

```text
checkpoints/mobilenetv3_small/
```

Il est aussi possible de logger le modèle complet dans MLflow avec :

```bash
python scripts/train_baseline.py \
    --epochs 1 \
    --batch-size 32 \
    --num-workers 0 \
    --experiment-name test_mlflow \
    --log-model
```

Cette option peut produire des artefacts plus volumineux.

---

## 8. Lancer un entraînement sur cluster Slurm

Le script Slurm principal est :

```text
scripts/job_script.sh
```

Exemple de test court :

```bash
sbatch scripts/job_script.sh --epochs 1 --batch-size 64 --num-workers 4
```

Exemple de run complet :

```bash
sbatch scripts/job_script.sh \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --experiment-name mobilenetv3_small_baseline
```

Le script Slurm demande actuellement :

```bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=30:00:00
```

---

## 9. Suivre un job Slurm

Lister les jobs actifs :

```bash
squeue -u $USER
```

Afficher l'état d'un job précis :

```bash
squeue -j <JOBID>
```

Afficher l'historique d'un job terminé :

```bash
sacct -j <JOBID>
```

Suivre la sortie standard :

```bash
tail -f slurm-datachallenge704-run001_<JOBID>.out
```

Suivre les warnings, erreurs et barres de progression `tqdm` :

```bash
tail -f slurm-datachallenge704-run001_<JOBID>.err
```

Annuler un job :

```bash
scancel <JOBID>
```

---

## 10. Fichiers générés après un run

Après chaque entraînement, le script génère automatiquement plusieurs fichiers.

### Soumission

```text
submissions/mobilenetv3_small_runXXX.csv
```

Exemple :

```text
submissions/mobilenetv3_small_run006.csv
```

### Checkpoint

```text
checkpoints/mobilenetv3_small/mobilenetv3_small_runXXX.pt
```

Exemple :

```text
checkpoints/mobilenetv3_small/mobilenetv3_small_run006.pt
```

### Log de run

```text
logs/mobilenetv3_small/mobilenetv3_small_runXXX.md
```

Exemple :

```text
logs/mobilenetv3_small/mobilenetv3_small_run006.md
```

Ce fichier contient notamment :

* le nom du run ;
* les hyperparamètres ;
* l'erreur train ;
* l'erreur validation ;
* les erreurs par genre ;
* la métrique finale de validation ;
* la durée du run ;
* le nom du checkpoint ;
* le nom du fichier de soumission.

### Artefacts MLflow

MLflow enregistre aussi les artefacts du run dans :

```text
mlartifacts/
```

Les artefacts incluent notamment :

* la soumission ;
* le checkpoint ;
* le log `.md` ;
* éventuellement le modèle complet si `--log-model` est utilisé.

### Logs Slurm

Lors d'un lancement avec `sbatch`, Slurm génère aussi :

```text
slurm-datachallenge704-run001_<JOBID>.out
slurm-datachallenge704-run001_<JOBID>.err
```

Ces fichiers peuvent ensuite être archivés dans :

```text
slurm_logs/
```

---

## 11. Rapatrier les résultats depuis le cluster avec `scp`

Les commandes suivantes sont à lancer depuis **l'ordinateur local**, pas depuis le cluster.

Créer d'abord un dossier local pour stocker les résultats :

```bash
mkdir cluster_results
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

Les checkpoints `.pt` peuvent être volumineux. Pour analyser un run, les fichiers les plus importants sont généralement :

* le `.csv` de soumission ;
* le `.md` de log ;
* le `.out` Slurm ;
* le `.err` Slurm ;
* les informations MLflow.

### Récupérer les fichiers MLflow

Pour récupérer la base MLflow et les artefacts depuis le cluster :

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlflow.db ./cluster_results/
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/mlartifacts ./cluster_results/
```

Ensuite, depuis le dossier `cluster_results` sur l'ordinateur local :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

### Récupérer tous les fichiers de soumission

```bash
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/submissions ./cluster_results/
```

### Récupérer tous les logs de run

```bash
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/logs ./cluster_results/
```

### Récupérer tous les logs Slurm

```bash
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_*.out ./cluster_results/
scp hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_*.err ./cluster_results/
```

### Récupérer tous les checkpoints

```bash
scp -r hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/checkpoints ./cluster_results/
```

---

## 12. Exemple de résultat obtenu

Un run de 100 epochs sur P100 a produit :

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

## 13. Reproductibilité

Les hyperparamètres principaux sont passés en ligne de commande :

```text
--epochs
--batch-size
--lr
--weight-decay
--num-workers
--experiment-name
--tracking-uri
--log-model
```

Ils sont automatiquement enregistrés :

* dans le fichier de log `.md` de chaque run ;
* dans MLflow.

Exemple local :

```bash
python scripts/train_baseline.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 0 \
    --experiment-name mobilenetv3_small_baseline
```

Sur cluster, la même configuration peut être lancée avec :

```bash
sbatch scripts/job_script.sh \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4 \
    --experiment-name mobilenetv3_small_baseline
```

---

## 14. Commandes utiles

### Tester rapidement le pipeline

```bash
python scripts/train_baseline.py --epochs 1 --batch-size 32 --num-workers 0
```

### Tester rapidement le pipeline avec MLflow

```bash
python scripts/train_baseline.py \
    --epochs 1 \
    --batch-size 32 \
    --num-workers 0 \
    --experiment-name test_mlflow
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
ls -lt logs/mobilenetv3_small/ | head
```

### Voir les dernières soumissions

```bash
ls -lt submissions/ | head
```

### Voir les derniers checkpoints

```bash
ls -lt checkpoints/mobilenetv3_small/ | head
```

### Lire le log d'un run

```bash
cat logs/mobilenetv3_small/mobilenetv3_small_run006.md
```

### Vérifier l'état du GPU

```bash
nvidia-smi
```

---

## 15. Remarques

Les barres de progression `tqdm` peuvent apparaître dans les fichiers `.err` Slurm. Ce n'est pas nécessairement une erreur.

Un warning CuDNN peut également apparaître au premier entraînement :

```text
Applied workaround for CuDNN issue, install nvrtc.so
```

Tant que l'entraînement se termine correctement et que les fichiers de sortie sont générés, ce warning peut être ignoré dans un premier temps.

La base MLflow `mlflow.db` et le dossier `mlartifacts/` ne doivent pas être poussés sur GitHub.

---

## 16. Pistes d'amélioration

Améliorations possibles pour la suite :

1. **Dégeler progressivement le backbone**
   Entraîner d'abord uniquement la tête, puis dégeler les derniers blocs de MobileNetV3.

2. **Ajouter de la data augmentation**
   Exemples :

   * horizontal flip ;
   * color jitter ;
   * random crop léger ;
   * brightness/contrast augmentation.

3. **Tester d'autres architectures**

   * EfficientNet ;
   * ConvNeXt ;
   * ResNet ;
   * MobileNetV3-Large.

4. **Ajouter un scheduler**
   Exemples :

   * `CosineAnnealingLR` ;
   * `ReduceLROnPlateau`.

5. **Améliorer la validation**

   * K-fold cross-validation ;
   * splits stratifiés plus robustes ;
   * analyse des erreurs par intervalle d'occlusion.

6. **Sauvegarder le meilleur checkpoint**
   Actuellement, le checkpoint final est sauvegardé. Une amélioration serait de sauvegarder le modèle avec la meilleure métrique validation.

7. **Ajouter un fichier de configuration**
   Utiliser un fichier YAML ou JSON pour enregistrer les paramètres de chaque expérience.

---

## 17. État actuel du projet

Le pipeline actuel permet de :

* charger les données ;
* entraîner une baseline MobileNetV3-Small ;
* évaluer sur validation avec la métrique pondérée par genre ;
* générer une soumission ;
* sauvegarder un checkpoint ;
* logger automatiquement les résultats ;
* suivre les expériences avec MLflow ;
* exécuter l'entraînement sur cluster Slurm avec GPU CUDA.

Ce projet constitue une base propre pour itérer sur les modèles et améliorer progressivement les performances.
