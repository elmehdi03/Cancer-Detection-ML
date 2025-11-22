# Models Directory

This directory contains trained machine learning models and preprocessing artifacts.

## Structure

```
models/
├── supervised/
│   ├── svm_model.pkl          # Trained SVM classifier
│   ├── xgboost_model.pkl      # Trained XGBoost classifier
│   └── scaler.pkl             # StandardScaler fitted on training data
├── unsupervised/
│   ├── isolation_forest.pkl   # Isolation Forest model
│   ├── lof_model.pkl          # Local Outlier Factor model
│   └── autoencoder.h5         # Autoencoder neural network
└── preprocessing/
    ├── pca_transformer.pkl    # PCA transformer (2 components)
    └── feature_names.pkl      # List of gene names used
```

## Usage

### Saving Models

Models are automatically saved after training in the notebook:
```python
import joblib
joblib.dump(model, 'models/supervised/svm_model.pkl')
```

### Loading Models

```python
import joblib
from tensorflow.keras.models import load_model

# Load supervised model
svm_model = joblib.load('models/supervised/svm_model.pkl')
scaler = joblib.load('models/supervised/scaler.pkl')

# Load autoencoder
autoencoder = load_model('models/unsupervised/autoencoder.h5')
```

## Notes

- Models are **not tracked by Git** (see `.gitignore`)
- Retrain models when data changes
- Models include version date in filename when saved with versioning
- Check model size before committing to ensure proper .gitignore rules

## Model Metadata

Track model performance in `models/model_registry.json` or add version tags when saving.
