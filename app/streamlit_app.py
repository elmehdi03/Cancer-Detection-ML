# Autres imports après config
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

# 🎨 Add custom CSS for styling and background
st.markdown("""
<style>
    /* Main background color - Medical/Scientific theme */
    .stApp {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%) !important;
    }
    
    /* Override default Streamlit background - fully opaque white */
    .main .block-container {
        background-color: #ffffff !important;
        padding: 2.5rem !important;
        border-radius: 15px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        margin-top: 2rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2b5876 0%, #4e4376 100%) !important;
    }
    
    /* Sidebar text - ensure high contrast */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Main content text - dark for readability */
    .main .block-container * {
        color: #1a1a1a !important;
    }
    
    /* Headers styling with gradient red effect */
    h1 {
        font-weight: 700 !important;
        font-size: 3rem !important;
    }
    
    h2 {
        color: #2a5298 !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #3d6bb3 !important;
        font-weight: 600 !important;
    }
    
    /* Paragraph text - pure black for maximum readability */
    p, span, div {
        color: #000000 !important;
    }
    
    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Info/Success/Warning boxes - ensure text is visible */
    .stAlert {
        border-radius: 8px !important;
        color: #000000 !important;
    }
    
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
    }
    
    /* File uploader area */
    [data-testid="stFileUploader"] {
        background-color: #f8f9fa !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border: 2px dashed #2a5298 !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #1e3c72 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1e3c72 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
    }
    
    .stButton>button:hover {
        background-color: #2a5298 !important;
    }
    
    /* Markdown text in main content */
    .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader {
        color: #1e3c72 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# 🧬 TITRE
st.markdown('<h1>🧬 <span style="background: linear-gradient(135deg, #e63946 0%, #f77f00 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">Détection du cancer du poumon par Machine Learning</span></h1>', unsafe_allow_html=True)
st.markdown("Analyse des profils d'expression génétique via méthodes supervisées et non supervisées.")

# 📥 Upload
uploaded_file = st.file_uploader("📂 Charger un fichier CSV (avec une colonne 'Label' pour le supervisé)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    df = df.dropna()
    
    # 📊 Display dataset info
    st.success(f"✅ Dataset loaded: {df.shape[0]} samples × {df.shape[1]} features")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - (1 if "Label" in df.columns else 0))
    with col3:
        if "Label" in df.columns:
            cancer_count = (df["Label"] == 1).sum()
            st.metric("Cancer samples", f"{cancer_count} ({cancer_count/len(df)*100:.1f}%)")
    
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
        
        # Show anomaly statistics
        anomaly_count = (preds == -1).sum()
        st.info(f"🔍 Detected {anomaly_count} anomalies out of {len(preds)} samples ({anomaly_count/len(preds)*100:.1f}%)")

        st.subheader("🔬 Projection PCA des anomalies")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="État prédit", 
                       palette={"Normal": "#2ecc71", "Anomalie": "#e74c3c"}, 
                       s=100, ax=ax, alpha=0.7)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.legend(title="État", loc="best")
        ax.grid(True, alpha=0.3)
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
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        st.subheader("📊 Performance Metrics")
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        col4.metric("F1-Score", f"{f1:.2%}")
        
        # Confusion matrix visualization
        st.subheader("📈 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Healthy', 'Cancer'],
                   yticklabels=['Healthy', 'Cancer'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_sup} Confusion Matrix')
        st.pyplot(fig)
        
        # Classification report
        with st.expander("📋 Detailed Classification Report"):
            st.text(classification_report(y_test, y_pred, target_names=['Healthy', 'Cancer']))
        
        # Feature importance for XGBoost
        if model_sup == "XGBoost":
            st.subheader("🧬 Top 10 Most Important Features")
            importances = clf.feature_importances_
            feature_names = X.columns
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis', ax=ax)
            ax.set_title('Top 10 Most Discriminative Features')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)

        df_pca["Classe prédite"] = clf.predict(X_scaled)
        df_pca["Classe prédite"] = df_pca["Classe prédite"].map({0: "Sain", 1: "Cancer"})

        st.subheader("📌 Projection PCA des classes prédites")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Classe prédite", 
                       palette={"Sain": "#3498db", "Cancer": "#e74c3c"}, 
                       s=100, ax=ax, alpha=0.7)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.legend(title="Predicted Class", loc="best")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    elif mode == "Supervisé":
        st.warning("❗ Le fichier chargé ne contient pas la colonne 'Label' nécessaire au modèle supervisé.")
else:
    st.info("👆 Veuillez charger un fichier CSV pour commencer l'analyse.")
    st.markdown("""
    ### 📖 Guide d'utilisation
    
    1. **Préparez vos données** : Format CSV avec les gènes en colonnes
    2. **Ajoutez une colonne 'Label'** pour l'analyse supervisée (0 = Sain, 1 = Cancer)
    3. **Chargez le fichier** via le bouton ci-dessus
    4. **Choisissez votre méthode** d'analyse dans la barre latérale
    
    ### 📊 Exemple de structure de données
    
    ```
    ,Gene1,Gene2,Gene3,...,Label
    Sample1,2.5,3.1,1.8,...,0
    Sample2,5.2,4.3,3.9,...,1
    ```
    
    Pour tester l'application, utilisez le fichier `expression_labelled.csv` disponible dans le dossier `data/`.
    """)
