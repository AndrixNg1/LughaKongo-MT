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
