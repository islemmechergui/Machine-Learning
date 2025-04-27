import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay  # Ajout de ConfusionMatrixDisplay

# 1. Charger les données
df = pd.read_csv("cleaned_data.csv", sep=';')

# 2. Préparer X et y
target_col = 'Total_Water_Consumption_Billion_Cubic_Meters'
median_val = df[target_col].median()
df['High_Consumption'] = (df[target_col] > median_val).astype(int)
features = [col for col in df.columns if col not in ['Country', 'Year', target_col, 'High_Consumption']]
features = [col for col in features if df[col].dtype in ['float64', 'int64']]
X = df[features]
y = df['High_Consumption']

# 3. Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Créer un modèle d'arbre de décision
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Visualiser l'arbre
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=features, class_names=["Low", "High"], filled=True)
plt.title("Arbre de décision ")
plt.savefig('tree_default.png', dpi=300) 
plt.show() 



# 6. Prédictions
y_pred = model.predict(X_test)
print("Prédictions pour les 5 premiers patients:", y_pred[:5])

# 7. Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 8. Validation croisée
cv_scores = cross_val_score(model, X, y, cv=5)
print("Scores validation croisée:", cv_scores)
print("Moyenne des scores:", np.mean(cv_scores))

# 9. Réentraînement avec 'entropy' et max_depth=3
model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)
acc_entropy = accuracy_score(y_test, y_pred_entropy)
print("Accuracy (criterion='entropy', max_depth=3):", acc_entropy)

# 10. Visualiser l'arbre ajusté
plt.figure(figsize=(30, 20))
plot_tree(model_entropy, feature_names=features, class_names=["Low", "High"], filled=True)
plt.title("Arbre de décision (criterion='entropy', max_depth=3)")
plt.savefig('tree.png', dpi=300)  # dpi 300 pour haute résolution

plt.show()  # pour voir directement l'arbre


# 1. Matrice de confusion pour le modèle avec 'gini'
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "High"])
disp.plot(cmap='Blues')
plt.title("Matrice de confusion (criterion='gini')")

plt.show()

# 2. Matrice de confusion pour le modèle avec 'entropy'
cm_entropy = confusion_matrix(y_test, y_pred_entropy)
disp_entropy = ConfusionMatrixDisplay(confusion_matrix=cm_entropy, display_labels=["Low", "High"])
disp_entropy.plot(cmap='Blues')
plt.title("Matrice de confusion (criterion='entropy')")
plt.show()

# 3. Tracer les scores de validation croisée
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-')
plt.title("Scores de validation croisée")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.grid(True)
plt.ylim(0, 1)
plt.show()

# 4. Importance des variables (features)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.title("Importance des variables (criterion='gini')")
plt.tight_layout()
plt.show()

# 5. Importance des variables pour le modèle avec 'entropy'
importances_entropy = model_entropy.feature_importances_
indices_entropy = np.argsort(importances_entropy)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances_entropy)), importances_entropy[indices_entropy])
plt.xticks(range(len(importances_entropy)), [features[i] for i in indices_entropy], rotation=90)
plt.title("Importance des variables (criterion='entropy')")
plt.tight_layout()
plt.show()