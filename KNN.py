import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📂 2. Charger les données nettoyées avec le bon séparateur
df = pd.read_csv('cleaned_data.csv', sep=';')

# Vérifier les colonnes du fichier
print(df.columns)

# Mise à jour du nom de la colonne cible
consumption_col = 'Total_Water_Consumption_Billion_Cubic_Meters'

# Vérification de l'existence de la colonne
if consumption_col not in df.columns:
    raise ValueError(f"La colonne {consumption_col} n'existe pas. Vérifie les noms de colonnes.")

# Création de la cible binaire
median_cons = df[consumption_col].median()
df['High_Consumption'] = (df[consumption_col] > median_cons).astype(int)

# Sélection des features
features = [col for col in df.columns if col not in [consumption_col, 'High_Consumption', 'Country', 'Year']]
features = [col for col in features if df[col].dtype in ['float64', 'int64']]

# Séparation des variables X et y
X = df[features]
y = df['High_Consumption']



# 3. Séparation train-test
print("\n=== Étape 3: Séparation des données ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Taille train: {X_train.shape}, Taille test: {X_test.shape}")

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Création et évaluation du modèle KNN avec K=3
print("\n=== Question 4: Modèle KNN avec K=3 ===")
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train_scaled, y_train)
y_pred_k3 = knn3.predict(X_test_scaled)

print("Prédictions pour les 5 premiers patients:", y_pred_k3[:5])
print("Accuracy (K=3):", accuracy_score(y_test, y_pred_k3))

# 5. Validation croisée
print("\n=== Question 5: Validation croisée ===")
cv_scores = cross_val_score(knn3, X_train_scaled, y_train, cv=5)
print("Scores de validation croisée:", cv_scores)
print("Moyenne des scores:", np.mean(cv_scores))

# 6. Optimisation de K avec GridSearchCV
print("\n=== Questions 6-7: Optimisation de K ===")
param_grid = {'n_neighbors': range(1, 26)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Meilleur modèle
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test_scaled)

print("Meilleur K:", grid_search.best_params_['n_neighbors'])
print("Meilleure accuracy:", grid_search.best_score_)

# 7. Visualisation des performances
print("\n=== Question 7: Visualisation ===")
plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'],
         grid_search.cv_results_['mean_test_score'],
         marker='o')
plt.xlabel('Nombre de voisins (K)')
plt.ylabel('Accuracy')
plt.title('Performance du KNN en fonction de K')
plt.grid()
plt.show()

# 8. Évaluation finale
print("\n=== Question 8: Évaluation finale ===")
print("Matrice de confusion:")
print(confusion_matrix(y_test, y_pred_best))
print("\nRapport de classification:")
print(classification_report(y_test, y_pred_best))

# Visualisation des données
print("\n=== Visualisation supplémentaire ===")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='High_Consumption', data=df)
plt.title('Distribution des classes')

plt.subplot(1, 2, 2)
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matrice de corrélation')

plt.tight_layout()
plt.show()