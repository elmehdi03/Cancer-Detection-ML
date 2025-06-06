# 🧬 Détection du Cancer du Poumon par Apprentissage Automatique

Projet de fin d’année réalisé dans le cadre du cursus UEMF-EIDIA, visant à détecter les cas de cancer pulmonaire à partir de données d'expression génétique.

---

## 🎯 Objectif

Utiliser des techniques d’apprentissage automatique (supervisées et non supervisées) pour distinguer les tissus cancéreux des tissus sains, en exploitant un dataset de type RNA-seq provenant de la base GEO.

---

## 🧪 Méthodologie

- **Prétraitement des données** : nettoyage, gestion des valeurs manquantes, standardisation
- **Réduction de dimension** : PCA
- **Méthodes non supervisées** : Isolation Forest, Local Outlier Factor, Autoencodeur
- **Méthodes supervisées** : SVM linéaire, XGBoost
- **Évaluation des performances** : accuracy, recall, precision, F1-score

---

## 📊 Résultats clés

- **XGBoost & SVM** atteignent une précision de **88 %** sur le jeu de test
- Les méthodes non supervisées donnent des résultats prometteurs mais moins robustes
- Visualisations projetées via PCA pour interprétation

---

## 📁 Données

Les données d'expression génique utilisées dans ce projet proviennent de l'étude [GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804).

Elles sont aussi disponibles dans ce dépôt, sous forme compressée : `GSE19804_series_matrix.txt.gz`.

Pour l'utiliser :

- Sous Windows : clic droit > Extraire
- En terminal : `gunzip GSE19804_series_matrix.txt.gz`

Une fois extrait, le fichier `GSE19804_series_matrix.txt` pourra être utilisé directement dans le notebook.

---

## 🚀 Application Streamlit

Ce projet inclut également une interface interactive développée avec **Streamlit** pour tester les modèles de classification et visualiser les résultats.

📄 Le fichier Streamlit est disponible ici : `streamlit_app_final.py`

📄 Le fichier utilisé en app streamlitest disponible en format zip sous le nom : `expression_labelled.zip`

➡️ Pour lancer l'application :

```bash
streamlit run streamlit_app_final.py


