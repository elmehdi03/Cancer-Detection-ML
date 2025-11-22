# Data Directory

This directory contains the dataset files for the Cancer Detection ML project.

## 📥 Required Files

The following data files are **not included in Git** due to their large size. You need to download them separately:

### 1. **GSE19804_series_matrix.txt** (~50 MB)
Raw gene expression data from GEO database.

**Download from:**
- **Primary source**: [NCBI GEO - GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804)
- Click "Download family" → "Series Matrix File(s)"
- Extract the `.gz` file to get `GSE19804_series_matrix.txt`
- Place in this `data/` directory

**Direct link**: [ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19804/matrix/](ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19804/matrix/)

### 2. **expression_labelled.csv** (~50 MB)
Processed data with cancer/healthy labels (generated from notebook).

This file is created by running the notebook `cancer_detection_pipeline.ipynb` or can be generated using the script:
```bash
python scripts/add_binary_label.py
```

## 📊 Data Description

**Study**: [GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804)
- **Title**: Gene expression profiling of lung cancer tissues
- **Organism**: Homo sapiens
- **Technology**: Microarray (Affymetrix)
- **Samples**: 60 lung cancer tissues + 60 normal tissues
- **Platform**: GPL570 [HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array

## 🚀 Quick Start

1. Download `GSE19804_series_matrix.txt` from NCBI GEO
2. Extract and place it in this `data/` directory
3. Run the notebook to generate `expression_labelled.csv`

## ⚠️ Note

These files are excluded from Git tracking via `.gitignore` to keep the repository lightweight. Each user must download them independently.
