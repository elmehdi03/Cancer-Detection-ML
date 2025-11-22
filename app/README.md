# App Directory

This directory contains the Streamlit application for interactive model testing.

## Files

- `streamlit_app.py` - Main Streamlit application

## Running the App

### First time setup:
```bash
# Make sure you've installed dependencies
pip install -r requirements.txt
```

### Run the app:
```bash
# From project root:
streamlit run app/streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Features

### Non-supervised Methods:
- **Isolation Forest** - Anomaly detection
- **Local Outlier Factor (LOF)** - Density-based outlier detection
- **Autoencoder** - Neural network-based anomaly detection

### Supervised Methods:
- **SVM** - Support Vector Machine classifier
- **XGBoost** - Gradient boosting classifier

### Visualizations:
- PCA projection of predictions
- Confusion matrices
- Classification reports

## Usage

1. **Upload your data**: CSV file with genes as columns
2. **For supervised learning**: Include a `Label` column (0 = Healthy, 1 = Cancer)
3. **Select analysis type**: Choose between supervised or unsupervised methods
4. **View results**: Visualizations and metrics will appear automatically

## Test Data

Use `../data/expression_labelled.csv` to test the app (includes Label column for supervised methods).

## Deployment

To deploy on Streamlit Cloud:
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository and select `app/streamlit_app.py`
4. Deploy!

**Note**: For deployment, you may need to:
- Create a `.streamlit/config.toml` file
- Ensure data files are accessible (or use file upload only)
- Check memory limits for large datasets
