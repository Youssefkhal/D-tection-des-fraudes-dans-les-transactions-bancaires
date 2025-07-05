import streamlit as st
import pandas as pd
import pickle
import io
from utils import convert_time_to_minutes

st.set_page_config(page_title="Détection de Fraude", layout="wide")

st.title("🔍 Analyse de Transactions (CSV)")

# Charger modèle et colonnes
try:
    model_pipeline = pickle.load(open("model_pipeline.pkl", "rb"))
    original_columns = pickle.load(open("original_columns.pkl", "rb"))
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# Upload CSV
uploaded_file = st.file_uploader("Chargez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        if not file_content.strip():
            st.error("Le fichier est vide.")
        else:
            df = pd.read_csv(io.StringIO(file_content))

            expected_cols = [
                'Montant', 'Heure', 'Type de paiement', 'Pays d\'origine', 
                'Appareil utilisé', 'Tentatives de connexion', 'Localisation IP'
            ]

            if not all(col in df.columns for col in expected_cols):
                missing = [col for col in expected_cols if col not in df.columns]
                st.error(f"Colonnes manquantes : {', '.join(missing)}")
            else:
                df['Heure_minutes'] = df['Heure'].apply(convert_time_to_minutes)
                df = df.drop('Heure', axis=1)
                processed_df = df[original_columns]

                predictions = model_pipeline.predict(processed_df)
                probas = model_pipeline.predict_proba(processed_df)

                df_result = pd.read_csv(io.StringIO(file_content))
                df_result["Prédiction"] = ["Fraude" if p == 1 else "Non-Fraude" for p in predictions]
                df_result["Confiance (%)"] = (probas.max(axis=1) * 100).round(2)

                st.success("✅ Prédictions générées")
                st.dataframe(df_result)

    except Exception as e:
        st.error(f"Erreur de traitement : {e}")
