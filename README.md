# 🧬 Détection du Cancer du Poumon par Apprentissage Automatique

Projet de fin d'année réalisé dans le cadre du cursus UEMF-EIDIA, visant à détecter les cas de cancer pulmonaire à partir de données d'expression génétique.

## 🎯 Objectif

Utiliser des techniques d'apprentissage automatique (supervisées et non supervisées) pour distinguer les tissus cancéreux des tissus sains, en exploitant un dataset de type RNA-seq provenant de la base GEO.

## 🧪 Méthodologie

- **Prétraitement des données** : nettoyage, gestion des valeurs manquantes, standardisation
- **Réduction de dimension** : PCA
- **Méthodes non supervisées** : Isolation Forest, Local Outlier Factor, Autoencodeur
- **Méthodes supervisées** : SVM linéaire, XGBoost
- **Évaluation des performances** : accuracy, recall, precision, F1-score

## 📊 Résultats clés

- **XGBoost** & **SVM** atteignent une précision de **88%** sur le jeu de test
- Les méthodes non supervisées donnent des résultats prometteurs mais moins robustes
- Visualisations projetées via PCA pour interprétation

## 📁 Structure du projet

```
Cancer-Detection-ML/
├── data/                           # Données brutes et traitées
│   ├── GSE19804_series_matrix.txt  # Données d'expression génique
│   └── expression_labelled.csv     # Données avec labels
├── notebooks/                      # Notebooks Jupyter
│   └── cancer_detection_pipeline.ipynb
├── scripts/                        # Scripts Python
│   └── add_binary_label.py
├── app/                            # Application Streamlit
│   └── streamlit_app.py
├── requirements.txt                # Dépendances Python
├── .gitignore                      # Fichiers à ignorer
├── LICENSE                         # License MIT
└── README.md                       # Ce fichier
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip

### Étapes d'installation

1. Cloner le repository :
```bash
git clone https://github.com/elmehdi03/Cancer-Detection-ML.git
cd Cancer-Detection-ML
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📁 Données

Les données d'expression génique utilisées dans ce projet proviennent de l'étude [GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804).

**⚠️ Note importante** : Les fichiers de données ne sont pas inclus dans ce dépôt Git en raison de leur taille (~50 MB).

### Téléchargement des données

1. **Télécharger depuis NCBI GEO** :
   - Visiter [GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804)
   - Cliquer sur "Download family" → "Series Matrix File(s)"
   - Télécharger `GSE19804_series_matrix.txt.gz`

2. **Extraire le fichier** :
   - **Windows** : clic droit > Extraire tout
   - **Terminal** : `gunzip GSE19804_series_matrix.txt.gz`

3. **Placer dans le dossier** :
   - Copier `GSE19804_series_matrix.txt` dans le dossier `data/`

Une fois téléchargé et extrait, le fichier pourra être utilisé directement dans le notebook.

Voir [data/README.md](data/README.md) pour plus de détails.

## 🔬 Utilisation

### Exploration des données

Ouvrir le notebook principal :
```bash
jupyter notebook notebooks/cancer_detection_pipeline.ipynb
```

### Application Streamlit

Lancer l'application web interactive :
```bash
streamlit run app/streamlit_app.py
```

L'application permet de :
- Tester les modèles de classification
- Visualiser les résultats
- Analyser les prédictions

## 📈 Résultats des modèles

### Méthodes non supervisées

| Modèle               | Accuracy | Recall Cancer | Précision Cancer | F1-score Cancer |
|----------------------|----------|---------------|------------------|-----------------|
| **Isolation Forest** | 57%      | 17%           | 83%              | 28%             |
| **Autoencodeur**     | 42%      | 2%            | 8%               | 3%              |
| **LOF**              | 53%      | 13%           | 67%              | 22%             |

### Méthodes supervisées

| Métrique         | SVM Supervisé | XGBoost |
|------------------|---------------|---------|
| Recall Cancer    | **83%**       | **83%** |
| Précision Cancer | 91%           | 91%     |
| F1-score Cancer  | **87%**       | **87%** |
| Accuracy         | **88%**       | **88%** |

## 🛠️ Technologies utilisées

- **Python 3.8+**
- **Pandas** & **NumPy** : manipulation de données
- **Scikit-learn** : modèles ML et prétraitement
- **XGBoost** : classification supervisée
- **TensorFlow/Keras** : autoencodeur
- **Matplotlib** & **Seaborn** : visualisation
- **Streamlit** : application web interactive

## 👥 Auteurs

- **El Mehdi** - [elmehdi03](https://github.com/elmehdi03)

## 📄 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- UEMF-EIDIA pour le cadre académique
- Base de données GEO pour les données d'expression génique
- La communauté open-source pour les outils et bibliothèques utilisés

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.
