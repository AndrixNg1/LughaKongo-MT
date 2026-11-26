# LughaKongo-MT

Projet d'équipe pour le Kongo MT Challenge d'IndabaX DRC 2026.

Notre objectif est de créer un seul modèle capable de traduire entre le français et cinq langues congolaises : le havu, le ngombe, le shi, le taabwa et le yansi. La traduction doit fonctionner dans les deux sens, soit dix directions au total.

## Organisation

- `data/raw/` : fichiers originaux du défi.
- `data/processed/` : données préparées pour les expériences.
- `notebooks/` : notebooks pour explorer les données et entraîner le modèle.
- `scripts/` : futurs scripts du projet.
- `configs/` : paramètres des expériences.
- `outputs/` : modèles, résultats et soumissions générés.
- `docs/` : notes et documentation de l'équipe.

Le notebook de départ fourni se trouve dans `notebooks/`. Il devra être vérifié et adapté avant l'entraînement.

## Pour commencer

L'équipe peut utiliser Google Colab pour exécuter les notebooks avec un GPU. Les fichiers originaux dans `data/raw/` sont ignorés par Git. Les fichiers préparés dans `data/processed/` peuvent être partagés dans le dépôt privé ; les modèles et résultats dans `outputs/` restent exclus.

Le fichier de validation officiel n'est pas encore présent. Une validation provisoire a été créée en séparant les versets du jeu d'entraînement : les deux directions et les langues partageant une même référence restent dans le même ensemble.

## Baseline sur Google Colab

Depuis la racine du dépôt, avec un GPU activé et les fichiers `data/processed/` présents :

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/baseline_nllb.json
python scripts/predict_and_verify.py --config configs/baseline_nllb.json
```

Dans une cellule Colab, préfixer chaque commande par `!` après s'être placé à la racine du dépôt. Le modèle et ses checkpoints sont enregistrés dans `outputs/`, ainsi que `outputs/submission.csv`. Télécharger ou sauvegarder ce dossier sur Drive avant la fin de la session.

La configuration utilise NLLB-200 distilled 600M. Pour expérimenter avec `google/byt5-small` ou `google/mt5-small`, modifier `model_name` et définir `model_family` à `t5`, puis réentraîner avant la prédiction. Utiliser la même configuration pour l'entraînement et la prédiction ; les modèles et les checkpoints des expériences ne doivent pas être mélangés. Pour NLLB, cinq nouveaux jetons de langue sont appris pendant l'entraînement : leur qualité reste à mesurer.

La sélection du modèle se fait avec chrF++ sur la validation ; BLEU avec tokenisation par espaces est également affiché. Les scores locaux peuvent différer du classement officiel. La soumission est contrôlée et reprend les ID et leur ordre du modèle fourni. Une traduction vide interrompt le programme pour permettre une correction avant la soumission.
