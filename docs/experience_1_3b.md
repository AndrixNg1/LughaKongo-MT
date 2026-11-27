# Expérience NLLB 1,3B

Cette expérience entraîne un seul modèle `facebook/nllb-200-distilled-1.3B` avec les mêmes données `Train.csv` et `Val.csv` que la baseline 600M. La validation reste réservée à l'évaluation : comparer les deux modèles avec chrF++ global et par direction, sur le même fichier et avec le même décodage. Aucun score n'est garanti.

La configuration `configs/nllb_1_3b_l4.json` vise un L4 avec batch de 1, accumulation de 32, Adafactor et gradient checkpointing. Ce réglage réduit les besoins mémoire mais peut être lent ; il n'a pas encore été testé sur GPU et peut nécessiter plus de VRAM. Les sorties et checkpoints doivent être distincts de ceux du 600M. Ne pas interrompre l'expérience en cours avant son checkpoint uniquement pour lancer celle-ci.

Depuis la racine du dépôt, installer les dépendances et monter Drive. Vérifier d'abord que le modèle tient dans la mémoire avec une expérience séparée :

```bash
python scripts/train.py --config configs/nllb_1_3b_l4.json --output_dir /content/smoke_1_3b --resume_from_checkpoint none --smoke_test_steps 3
```

Un smoke test vérifie quelques pas d'entraînement puis génère deux traductions, mais ne vérifie ni la validation complète ni la qualité du modèle ; il ne produit pas de modèle final. Ensuite, pour entraîner et conserver les checkpoints et le meilleur modèle sur Drive :

```bash
python scripts/train.py --config configs/nllb_1_3b_l4.json --output_dir /content/drive/MyDrive/LughaKongo-MT/checkpoints_1_3b --model_dir /content/drive/MyDrive/LughaKongo-MT/best_model_1_3b --resume_from_checkpoint auto
```

Après la fin, générer une soumission avec ce modèle :

```bash
python scripts/predict_and_verify.py --config configs/nllb_1_3b_l4.json --model_dir /content/drive/MyDrive/LughaKongo-MT/best_model_1_3b --drive_backup_dir /content/drive/MyDrive/LughaKongo-MT/experiment_1_3b
```

Sur Colab, préfixer les commandes par `!`. Le notebook tiers observé affiche un BLEU de validation de 28,66 pour un checkpoint, mais les détails de cette validation ne sont pas connus et le chiffre n'est pas comparable directement à notre chrF++. Son ensemble à deux modèles n'est pas retenu : le défi impose un seul modèle.
