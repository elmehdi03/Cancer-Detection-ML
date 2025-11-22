import pandas as pd
import sys
from pathlib import Path

# Use relative path from scripts/ to data/
data_path = Path("../data/expression_clean.csv")

# Check if file exists
if not data_path.exists():
    print(f"Error: File not found at {data_path}")
    print("Please ensure expression_clean.csv is in the data/ directory")
    sys.exit(1)

# Charger ton DataFrame principal
try:
    df = pd.read_csv(data_path, index_col=0)
    df.index = df.index.astype(str).str.strip().str.replace('"', '')
    print(f"✓ Data loaded successfully: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Générer les étiquettes à partir des noms d'échantillons
sample_labels = {}
for idx in df.index:
    lower = idx.lower()
    if "normal" in lower:
        sample_labels[idx] = "Sain"
    elif "cancer" in lower or "tumor" in lower:
        sample_labels[idx] = "Cancer"

# Appliquer au DataFrame
df["Status"] = df.index.map(sample_labels)
df = df.dropna(subset=["Status"])
df["Label"] = df["Status"].map({"Sain": 0, "Cancer": 1})

# Vérification
print("✅ Étiquettes ajoutées avec succès. Répartition :")
print(df["Label"].value_counts())

# Sauvegarde
output_path = Path("../data/expression_labelled.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path)
print(f"✓ Labeled data saved to {output_path}")
