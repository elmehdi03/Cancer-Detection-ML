# Autres imports après config
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# 💡 Configuration Streamlit - toujours en 1er
st.set_page_config(
    page_title="Cancer du poumon - ML App",
    layout="wide",
    page_icon="🧬"
)

# 🎨 FOND
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0);  /* fond totalement transparent */
        padding: 2rem;
        margin: 2rem;
    }}
    h1, h2, h3, p, .stMarkdown, .stText, .stSubheader {{
        color: #00000t;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("BackGround_ML.png")

# 🧬 TITRE
st.title("🧬 Détection du cancer du poumon par Machine Learning")
st.markdown("Analyse des profils d'expression génétique via méthodes supervisées et non supervisées.")

# 📥 Upload
uploaded_file = st.file_uploader("📂 Charger un fichier CSV (avec une colonne 'Label' pour le supervisé)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    df = df.dropna()
    st.sidebar.header("⚙️ Paramètres")

    if "Label" in df.columns:
        has_labels = True
        X = df.drop(columns=["Status", "Label"], errors="ignore")
        y = df["Label"]
    else:
        has_labels = False
        X = df.copy()

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=df.index)

    # Choix du mode
    mode = st.sidebar.radio("🧠 Type d'analyse :", ["Non supervisé", "Supervisé"])

    # 🌐 NON SUPERVISÉ
    if mode == "Non supervisé":
        algo = st.sidebar.selectbox("🧪 Méthode non supervisée :", ["Isolation Forest", "Autoencodeur", "LOF"])

        if algo == "Isolation Forest":
            model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
            preds = model.fit_predict(X_scaled)

        elif algo == "LOF":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            preds = model.fit_predict(X_scaled)

        elif algo == "Autoencodeur":
            input_layer = Input(shape=(X_scaled.shape[1],))
            encoded = Dense(64, activation='relu')(input_layer)
            encoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(64, activation='relu')(encoded)
            output_layer = Dense(X_scaled.shape[1], activation='linear')(decoded)
            ae = Model(inputs=input_layer, outputs=output_layer)
            ae.compile(optimizer='adam', loss='mse')
            ae.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=0)
            reconstructed = ae.predict(X_scaled)
            errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
            threshold = np.percentile(errors, 90)
            preds = np.where(errors > threshold, -1, 1)

        df_pca["État prédit"] = np.where(preds == -1, "Anomalie", "Normal")

        st.subheader("🔬 Projection PCA des anomalies")
        fig, ax = plt.subplots(figsize=(5, 3.5))  # Plus petit, pour éviter qu'il prenne trop d’espace
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="État prédit", palette="Set1", ax=ax)
        st.pyplot(fig)
       

    # ✅ SUPERVISÉ
    elif mode == "Supervisé" and has_labels:
        model_sup = st.sidebar.selectbox("📈 Modèle supervisé :", ["SVM", "XGBoost"])
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        if model_sup == "SVM":
            clf = SVC(kernel="linear", probability=True)
        else:
            clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.subheader("📊 Évaluation du modèle")
        st.text("Matrice de confusion :")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Rapport de classification :")
        st.text(classification_report(y_test, y_pred))

        df_pca["Classe prédite"] = clf.predict(X_scaled)
        df_pca["Classe prédite"] = df_pca["Classe prédite"].map({0: "Sain", 1: "Cancer"})

        st.subheader("📌 Projection PCA des classes prédites")
        fig, ax = plt.subplots(figsize=(5, 3.5))  # Plus petit, pour éviter qu'il prenne trop d’espace
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Classe prédite", palette="coolwarm", ax=ax)
        st.pyplot(fig)

    elif mode == "Supervisé":
        st.warning("❗ Le fichier chargé ne contient pas la colonne 'Label' nécessaire au modèle supervisé.")
