import streamlit as st
import subprocess

st.title("🔄 Réentraîner le modèle")

st.warning("Ceci va relancer l'entraînement avec le fichier `data/transactions.csv`.")

if st.button("Lancer le réentraînement"):
    try:
        result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
        st.code(result.stdout)
        st.success("✅ Modèle réentraîné avec succès")
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement : {e}")
