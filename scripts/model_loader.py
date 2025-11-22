"""
Model Loading Utilities

This module provides functions to load trained models and preprocessing artifacts.
"""

import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np


class ModelLoader:
    """Load and manage trained models for cancer detection."""
    
    def __init__(self, models_dir='../models'):
        """
        Initialize model loader.
        
        Args:
            models_dir: Path to models directory
        """
        self.models_dir = Path(models_dir)
        self.supervised_dir = self.models_dir / 'supervised'
        self.unsupervised_dir = self.models_dir / 'unsupervised'
        self.preprocessing_dir = self.models_dir / 'preprocessing'
    
    def load_scaler(self):
        """Load the fitted StandardScaler."""
        scaler_path = self.preprocessing_dir / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        return joblib.load(scaler_path)
    
    def load_pca(self):
        """Load the fitted PCA transformer."""
        pca_path = self.preprocessing_dir / 'pca_transformer.pkl'
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA transformer not found at {pca_path}")
        return joblib.load(pca_path)
    
    def load_feature_names(self):
        """Load the list of feature (gene) names."""
        features_path = self.preprocessing_dir / 'feature_names.pkl'
        if not features_path.exists():
            raise FileNotFoundError(f"Feature names not found at {features_path}")
        return joblib.load(features_path)
    
    def load_svm(self):
        """Load the trained SVM model."""
        svm_path = self.supervised_dir / 'svm_model.pkl'
        if not svm_path.exists():
            raise FileNotFoundError(f"SVM model not found at {svm_path}")
        return joblib.load(svm_path)
    
    def load_xgboost(self):
        """Load the trained XGBoost model."""
        xgb_path = self.supervised_dir / 'xgboost_model.pkl'
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
        return joblib.load(xgb_path)
    
    def load_isolation_forest(self):
        """Load the trained Isolation Forest model."""
        if_path = self.unsupervised_dir / 'isolation_forest.pkl'
        if not if_path.exists():
            raise FileNotFoundError(f"Isolation Forest model not found at {if_path}")
        return joblib.load(if_path)
    
    def load_autoencoder(self):
        """Load the trained Autoencoder model."""
        ae_path = self.unsupervised_dir / 'autoencoder.h5'
        if not ae_path.exists():
            raise FileNotFoundError(f"Autoencoder model not found at {ae_path}")
        return load_model(ae_path)
    
    def preprocess_data(self, X_raw):
        """
        Apply preprocessing to raw data.
        
        Args:
            X_raw: Raw data (samples × genes)
        
        Returns:
            X_scaled: Standardized data
        """
        scaler = self.load_scaler()
        feature_names = self.load_feature_names()
        
        # Ensure features match
        if isinstance(X_raw, pd.DataFrame):
            missing_features = set(feature_names) - set(X_raw.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            X_raw = X_raw[feature_names]
        
        X_scaled = scaler.transform(X_raw)
        return X_scaled
    
    def predict_supervised(self, X_raw, model_name='svm'):
        """
        Make predictions using supervised model.
        
        Args:
            X_raw: Raw data (samples × genes)
            model_name: 'svm' or 'xgboost'
        
        Returns:
            predictions: Class predictions (0=Healthy, 1=Cancer)
        """
        # Preprocess
        X_scaled = self.preprocess_data(X_raw)
        
        # Load model
        if model_name.lower() == 'svm':
            model = self.load_svm()
        elif model_name.lower() == 'xgboost':
            model = self.load_xgboost()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Predict
        predictions = model.predict(X_scaled)
        return predictions
    
    def predict_anomaly(self, X_raw, method='isolation_forest'):
        """
        Detect anomalies using unsupervised model.
        
        Args:
            X_raw: Raw data (samples × genes)
            method: 'isolation_forest' or 'autoencoder'
        
        Returns:
            anomalies: -1 for anomaly, 1 for normal
        """
        # Preprocess
        X_scaled = self.preprocess_data(X_raw)
        
        if method.lower() == 'isolation_forest':
            model = self.load_isolation_forest()
            anomalies = model.predict(X_scaled)
        
        elif method.lower() == 'autoencoder':
            autoencoder = self.load_autoencoder()
            reconstructed = autoencoder.predict(X_scaled)
            errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
            # Use 90th percentile as threshold
            threshold = np.percentile(errors, 90)
            anomalies = np.where(errors > threshold, -1, 1)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return anomalies


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = ModelLoader()
    
    # Load preprocessing artifacts
    print("Loading preprocessing artifacts...")
    scaler = loader.load_scaler()
    pca = loader.load_pca()
    feature_names = loader.load_feature_names()
    print(f"✓ Loaded {len(feature_names)} features")
    
    # Load models
    print("\nLoading models...")
    svm_model = loader.load_svm()
    print("✓ Loaded SVM model")
    
    xgb_model = loader.load_xgboost()
    print("✓ Loaded XGBoost model")
    
    iso_forest = loader.load_isolation_forest()
    print("✓ Loaded Isolation Forest model")
    
    autoencoder = loader.load_autoencoder()
    print("✓ Loaded Autoencoder model")
    
    print("\n✅ All models loaded successfully!")
