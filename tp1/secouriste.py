import pandas as pd

# Chargement des donnees
df = pd.read_csv("titanic.csv", sep=';')

# Extraction des colonnes pertinentes et convertion des chaines de caracteres en entier
df = df[["poids", "taille"]]
df["taille"] = df["taille"].map({"petit": 0, "grand": 1})

# Suppression des donnees incompletes
df = df.dropna()

# Construire arbre de décision ...


# Afficher la matrice de confusion ...