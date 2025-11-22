# Data Directory

This directory contains the dataset files for the Cancer Detection ML project.

## Files

- `GSE19804_series_matrix.txt` - Raw gene expression data from GEO database
- `expression_labelled.csv` - Processed data with cancer/healthy labels

## Data Source

The data comes from the [GSE19804](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19804) study:
- **Title**: Gene expression profiling of lung cancer tissues
- **Organism**: Homo sapiens
- **Technology**: RNA-seq

## Usage

Do not commit large data files to Git. Add them to `.gitignore` instead.

For sharing:
- Use Git LFS for large files
- Or provide download links in the README
- Or host on external platforms (Kaggle, Google Drive, etc.)
