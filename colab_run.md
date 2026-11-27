# Entraîner sur Google Colab

Dans Colab, choisir **Exécution > Modifier le type d'exécution > GPU**. Ces cellules utilisent la branche `test` : pousser d'abord les changements vers cette branche et adapter si nécessaire. Pour un dépôt privé, authentifier l'accès GitHub dans Colab ; ne pas afficher de jeton dans le notebook. Vérifier que les quatre CSV sont présents dans `data/processed/` après le clonage.

## GPU et Google Drive

```python
!nvidia-smi
from google.colab import drive
drive.mount('/content/drive')
```

## Dépôt et dépendances

Dans une nouvelle session :

```python
%cd /content
!git clone --branch test https://github.com/AndrixNg1/LughaKongo-MT.git
%cd /content/LughaKongo-MT
!pip install -q -r requirements.txt
```

Si le dépôt est déjà cloné, se placer à sa racine et récupérer les changements sans écraser de travail local. Si Colab demande de redémarrer après installation, redémarrer puis remonter Drive et retourner dans le dossier du dépôt.

## Hugging Face : seulement si nécessaire

Le modèle public ne nécessite normalement pas de jeton. Si un accès authentifié est demandé, enregistrer le secret `HF_TOKEN` dans les secrets Colab puis exécuter :

```python
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'), add_to_git_credential=False)
```

## Entraînement et reprise

```python
!python scripts/train.py --config configs/nllb_tuned.json --output_dir /content/drive/MyDrive/LughaKongo-MT/checkpoints --model_dir /content/drive/MyDrive/LughaKongo-MT/best_model --resume_from_checkpoint auto
```

La commande reprend le dernier checkpoint complet trouvé ou démarre un nouvel entraînement. Garder les mêmes données, configuration et versions pour reprendre ; pour une expérience différente, choisir de nouveaux dossiers. Une interruption peut faire perdre le travail depuis le dernier checkpoint ; attendre la fin de l'évaluation et de la sauvegarde. Chaque évaluation traduit l'ensemble de `Val.csv`, ce qui peut prendre du temps. La sortie finale du meilleur modèle est copiée dans `best_model` à la fin de l'entraînement.

## Soumission

Après la fin de l'entraînement :

```python
!python scripts/predict_and_verify.py --config configs/nllb_tuned.json --model_dir /content/drive/MyDrive/LughaKongo-MT/best_model --drive_backup_dir /content/drive/MyDrive/LughaKongo-MT
```

Le fichier vérifié est `outputs/tuned/submission.csv` et sa copie est `/content/drive/MyDrive/LughaKongo-MT/submission.csv`. Le script interrompt la génération en cas d'ID incohérent ou de traduction vide.
