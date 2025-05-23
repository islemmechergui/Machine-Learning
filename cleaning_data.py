"""import pandas as pd
import numpy as np

# 1. Chargement et inspection des données
print("=== Étape 1: Chargement et inspection des données ===")
file_path = 'cleaned_global_water_consumption.csv'  # Assurez-vous que le fichier existe
data = pd.read_csv(file_path, sep=';')
print("=== Premières lignes ===")
print(data.head())
print("\n=== Infos sur les colonnes ===")
print(data.info())
print("\n=== Noms des colonnes ===")
print(data.columns.tolist())

# 2. Nettoyage des données
print("\n=== Étape 2: Nettoyage des données ===")
# Nettoyage des noms de colonnes
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(' ', '_')
data.columns = data.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)

# Gestion des valeurs manquantes
for col in data.columns:
    if data[col].dtype in ['float64', 'int64']:
        data[col].fillna(data[col].median(), inplace=True)

print("\nDonnées nettoyées:")
print(data.head())

# Sauvegarde des données nettoyées dans un fichier CSV
print("\n=== Étape 3: Sauvegarde des données nettoyées ===")
cleaned_file_path = 'cleaned_data.csv'
data.to_csv(cleaned_file_path, index=False, sep=';')  # Sauvegarde avec séparateur ';'
print(f"Données nettoyées sauvegardées dans le fichier: {cleaned_file_path}")


"""



import pandas as pd
import numpy as np

# 1. Chargement et inspection des données
print("=== Étape 1: Chargement et inspection des données ===")
file_path = 'cleaned_global_water_consumption.csv'  # Assurez-vous que le fichier existe
try:
    data = pd.read_csv(file_path, sep=';')
except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' est introuvable.")
    exit()
except Exception as e:
    print(f"Une erreur s'est produite lors du chargement du fichier : {e}")
    exit()

print("=== Premières lignes ===")
print(data.head())

print("\n=== Infos sur les colonnes ===")
print(data.info())

print("\n=== Noms des colonnes ===")
print(data.columns.tolist())

# 2. Nettoyage des données
print("\n=== Étape 2: Nettoyage des données ===")

# Nettoyage des noms de colonnes
data.columns = data.columns.str.strip()  # Supprime les espaces au début et à la fin
data.columns = data.columns.str.replace(' ', '_')  # Remplace les espaces par des underscores
data.columns = data.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)  # Supprime les caractères spéciaux

# Gestion des valeurs manquantes
for col in data.columns:
    if data[col].dtype in ['float64', 'int64']:  # Vérifie si la colonne est numérique
        data[col] = data[col].fillna(data[col].median())

print("\nDonnées nettoyées:")
print(data.head())

# 3. Sauvegarde des données nettoyées dans un fichier CSV
print("\n=== Étape 3: Sauvegarde des données nettoyées ===")
cleaned_file_path = 'cleaned_data.csv'
try:
    data.to_csv(cleaned_file_path, index=False, sep=';')  # Sauvegarde avec séparateur ';'
    print(f"Données nettoyées sauvegardées dans le fichier: {cleaned_file_path}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde du fichier : {e}")



   