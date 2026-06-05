# Data Challenge 704 — Face Occlusion Prediction

Ce projet a été réalisé dans le cadre du **Data Challenge 704**.
L'objectif est de prédire le niveau d'occlusion d'un visage à partir d'images prétraitées, en tenant compte d'une métrique finale pondérée et équilibrée entre les genres.

Le pipeline actuel entraîne un modèle de régression basé sur **MobileNetV3-Small pré-entraîné sur ImageNet**, puis génère automatiquement :

* une soumission `.csv` ;
* un checkpoint PyTorch `.pt` ;
* un fichier de log `.md` contenant les métriques du run ;
* les logs Slurm `.out` et `.err` lors de l'exécution sur cluster.

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
├── notebooks/                   # Notebooks exploratoires
│
├── occlusion_datasets/          # Fichiers CSV train/test
│   ├── train.csv
│   └── test_students.csv
│
├── scripts/
│   ├── train_baseline.py        # Point d'entrée principal pour l'entraînement
│   └── job_script.sh            # Script Slurm pour le cluster
│
├── src/
│   ├── config.py                # Chemins et constantes globales
│   ├── data.py                  # Chargement des données et DataLoaders
│   ├── datasets.py              # Dataset PyTorch
│   ├── engine.py                # Boucle d'entraînement
│   ├── losses.py                # Fonction de loss
│   ├── metrics.py               # Métriques de validation
│   ├── models.py                # Construction du modèle
│   ├── predict.py               # Fonctions de prédiction
│   └── utils.py                 # Fonctions utilitaires : logs, checkpoints, run id
│
├── submissions/                 # Fichiers CSV de soumission
│
├── requirements.txt             # Dépendances Python
├── README.md
└── task_brief.pdf               # Sujet du challenge
```

---

## 2. Objectif du challenge

Pour chaque image de visage, le modèle doit prédire une valeur continue :

```text
FaceOcclusion ∈ [0, 1]
```

Cette valeur représente le niveau d'occlusion du visage.

Le problème est traité comme une tâche de **régression supervisée**.

---

## 3. Données utilisées

Les données principales se trouvent dans :

```text
occlusion_datasets/train.csv
occlusion_datasets/test_students.csv
```

Le dossier d'images utilisé est :

```text
crops/Crop_224_5fp_100K/
```

Le fichier `train.csv` contient notamment :

* `filename` : nom du fichier image ;
* `FaceOcclusion` : cible de régression ;
* `gender` : variable utilisée dans la métrique finale.

Le fichier `test_students.csv` contient les images pour lesquelles une prédiction doit être générée.

---

## 4. Modèle utilisé

Le modèle de base est :

```text
MobileNetV3-Small
```

Il est chargé avec les poids pré-entraînés ImageNet :

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

## 5. Loss et métrique

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

## 6. Installation

Depuis la racine du projet :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pour vérifier que PyTorch voit bien le GPU CUDA :

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

---

## 7. Lancer un entraînement en local

Pour tester rapidement le pipeline :

```bash
python scripts/train_baseline.py --epochs 1 --batch-size 32 --num-workers 0
```

Pour un entraînement plus long :

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
    --num-workers 4
```

Le script demande actuellement :

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

Après chaque entraînement, le script génère automatiquement :

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
* le nom du checkpoint ;
* le nom du fichier de soumission.

---

## 11. Rapatrier les résultats depuis le cluster

Depuis l'ordinateur local, utiliser `rsync` :

```bash
mkdir -p cluster_results

rsync -avz hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/submissions/ ./cluster_results/submissions/
rsync -avz hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/logs/ ./cluster_results/logs/
rsync -avz 'hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_*.out' ./cluster_results/
rsync -avz 'hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/slurm-datachallenge704-run001_*.err' ./cluster_results/
```

Pour récupérer aussi les checkpoints :

```bash
rsync -avz hamon-25@gpu-gw:/home/infres/hamon-25/datachallenge704/checkpoints/ ./cluster_results/checkpoints/
```

Les checkpoints peuvent être volumineux. Pour analyser les résultats, les fichiers `.csv`, `.md`, `.out` et `.err` suffisent souvent.

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

```bash
--epochs
--batch-size
--lr
--weight-decay
--num-workers
```

Ils sont automatiquement enregistrés dans le fichier de log `.md` de chaque run.

Exemple :

```bash
python scripts/train_baseline.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4
```

---

## 14. Pistes d'amélioration

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
   Exemple :

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

## 15. Commandes utiles

### Tester rapidement le pipeline

```bash
python scripts/train_baseline.py --epochs 1 --batch-size 32 --num-workers 0
```

### Lancer un run cluster

```bash
sbatch scripts/job_script.sh --epochs 100 --batch-size 128 --num-workers 4
```

### Voir les derniers logs

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

---

## 16. Remarques

Les barres de progression `tqdm` peuvent apparaître dans les fichiers `.err` Slurm. Ce n'est pas nécessairement une erreur.

Un warning CuDNN peut également apparaître au premier entraînement :

```text
Applied workaround for CuDNN issue, install nvrtc.so
```

Tant que l'entraînement se termine correctement et que les fichiers de sortie sont générés, ce warning peut être ignoré dans un premier temps.

---

## 17. État actuel du projet

Le pipeline actuel permet de :

* charger les données ;
* entraîner une baseline MobileNetV3-Small ;
* évaluer sur validation avec la métrique pondérée par genre ;
* générer une soumission ;
* sauvegarder un checkpoint ;
* logger automatiquement les résultats ;
* exécuter l'entraînement sur cluster Slurm avec GPU CUDA.

Ce projet constitue une base propre pour itérer sur les modèles et améliorer progressivement les performances.
