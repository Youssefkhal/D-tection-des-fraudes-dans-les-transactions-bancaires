import streamlit as st
import pandas as pd
import os
import pickle
from utils import convert_time_to_minutes

st.title("➕ Ajouter une Nouvelle Transaction & Détecter la Fraude")

# Charger le modèle
try:
    model_pipeline = pickle.load(open("model_pipeline.pkl", "rb"))
    original_columns = pickle.load(open("original_columns.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Erreur de chargement du modèle : {e}")
    st.stop()

with st.form("form"):
    montant = st.text_input("Montant", "0.0")
    heure = st.text_input("Heure (HH:MM)", "13:45")
    type_paiement = st.text_input("Type de paiement", "PAYMENT")
    pays = st.text_input("Pays d'origine", "FR")
    appareil = st.text_input("Appareil utilisé", "Mobile")
    tentatives = st.text_input("Tentatives de connexion", "0")
    ip = st.text_input("Localisation IP", "192.168.1.1")

    submitted = st.form_submit_button("Analyser et Ajouter")

if submitted:
    try:
        # Conversion des types
        montant = float(montant)
        tentatives = int(tentatives)
        heure_min = convert_time_to_minutes(heure)

        # Créer DataFrame pour prédiction
        input_data = pd.DataFrame([{
            'Montant': montant,
            'Heure_minutes': heure_min,
            'Type de paiement': type_paiement,
            'Pays d\'origine': pays,
            'Appareil utilisé': appareil,
            'Tentatives de connexion': tentatives,
            'Localisation IP': ip
        }])

        # Réordonner les colonnes comme attendues
        input_data = input_data[original_columns]

        # Prédiction
        prediction = model_pipeline.predict(input_data)[0]
        proba = model_pipeline.predict_proba(input_data)[0].max()

        is_fraud = int(prediction)
        label = "🛑 FRAUDE" if is_fraud else "✅ Non-fraude"
        confidence = round(proba * 100, 2)

        st.subheader("Résultat de la prédiction :")
        st.markdown(f"**Prédiction :** {label}")
        st.markdown(f"**Confiance :** {confidence}%")

        # Ajouter la transaction au fichier CSV
        new_data = {
            'Montant': montant,
            'Heure_minutes': heure_min,
            'Type de paiement': type_paiement,
            'Pays d\'origine': pays,
            'Appareil utilisé': appareil,
            'Tentatives de connexion': tentatives,
            'Localisation IP': ip,
            'is_fraud': is_fraud
        }

        os.makedirs("data", exist_ok=True)
        csv_path = "data/transactions.csv"

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        else:
            df = pd.DataFrame([new_data])

        df.to_csv(csv_path, index=False)
        st.success("💾 Transaction enregistrée avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
