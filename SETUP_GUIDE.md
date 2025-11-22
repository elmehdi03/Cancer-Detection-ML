# Project Setup Guide

## 📂 Reorganize Files

To organize your project properly, please move files to the following locations:

### Data files → `data/` folder
```powershell
Move-Item -Path ".\expression_labelled.csv" -Destination ".\data\"
Move-Item -Path ".\GSE19804_series_matrix.txt" -Destination ".\data\"
Move-Item -Path ".\expression_labelled.zip" -Destination ".\data\"
```

### Notebooks → `notebooks/` folder
```powershell
Move-Item -Path ".\NoteBooks\01_exploration_GSE19804.ipynb" -Destination ".\notebooks\"
```

### Scripts → `scripts/` folder
```powershell
Move-Item -Path ".\NoteBooks\ajouter_label_binaire.py" -Destination ".\scripts\"
```

### Remove old folder
```powershell
Remove-Item -Path ".\NoteBooks" -Recurse -Force
```

## 🚀 Next Steps

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Update file paths in notebooks:**
   - Open `notebooks/01_exploration_GSE19804.ipynb`
   - Update data paths from `C:/Users/...` to relative paths like `../data/GSE19804_series_matrix.txt`

3. **Update file paths in scripts:**
   - Open `scripts/ajouter_label_binaire.py`
   - Update paths to use relative paths

4. **Initialize Git (if not already done):**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: project structure and documentation"
   ```

5. **Push to GitHub:**
   ```powershell
   git remote add origin https://github.com/elmehdi03/Cancer-Detection-ML.git
   git branch -M main
   git push -u origin main
   ```

## 📝 Additional Improvements

Consider adding:
- Unit tests in a `tests/` folder
- Configuration file for hyperparameters
- Model serialization/loading utilities
- Data validation scripts
- Documentation in a `docs/` folder
