import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 🔹 Chargement des données
try:
    df = pd.read_csv('fraud_dataset.csv')
except FileNotFoundError:
    print("❌ Erreur : Le fichier 'fraud_dataset.csv' est introuvable.")
    exit()

# 🔹 Séparation des variables
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# 🔹 Définir les colonnes numériques et catégorielles
numerical_features = ['Montant', 'Tentatives de connexion']
categorical_features = ['Type de paiement', 'Pays d\'origine', 'Appareil utilisé', 'Localisation IP']

# 🔹 Conversion de la colonne Heure en minutes
def convert_time_to_minutes(time_str):
    if pd.isna(time_str):
        return -1
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except ValueError:
        return -1

X['Heure_minutes'] = X['Heure'].apply(convert_time_to_minutes)
numerical_features.append('Heure_minutes')
X = X.drop('Heure', axis=1)

# 🔹 Préprocesseur
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 🔹 Pipeline complet
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 🔹 Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Entraînement
model_pipeline.fit(X_train, y_train)

# 🔹 Prédictions
y_pred = model_pipeline.predict(X_test)

# 🔹 Évaluation
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 🔹 Affichage terminal
print("\n📊 Évaluation du modèle :")
print("Accuracy :", accuracy)
print("\nClassification Report :\n", class_report)
print("Confusion Matrix :\n", conf_matrix)

# 🔹 Sauvegarde des métriques dans un fichier texte
with open("model_metrics.txt", "w") as f:
    f.write(" Évaluation du modèle\n")
    f.write(f"Accuracy : {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report + "\n")
    f.write("Confusion Matrix:\n")
    for row in conf_matrix:
        f.write(" ".join(map(str, row)) + "\n")

print("✅ Métriques sauvegardées dans 'model_metrics.txt'")

# 🔹 Sauvegarde du pipeline
with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# 🔹 Sauvegarde des noms de colonnes
with open('original_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Pipeline et colonnes sauvegardés dans 'model_pipeline.pkl' et 'original_columns.pkl'")
