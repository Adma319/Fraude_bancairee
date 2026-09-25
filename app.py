import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Chargement du modèle fraude bancaire

model_pipeline = joblib.load('modele_fraude_bancaire.pkl')

st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title('💳 Application de Détection de Fraude Bancaire')
st.write('Entrez les caractéristiques d\'une transaction pour prédire si elle est frauduleuse.')


# Crée des champs d'entrée pour les caractéristiques
def get_user_input():
    Time = st.sidebar.number_input('Time (secondes écoulées depuis la 1ère transaction)', 0.0, 180000.0, 1000.0, 1.0)
    Amount = st.sidebar.number_input('Amount (montant de la transaction)', 0.0, 26000.0, 100.0, 0.01)

    st.sidebar.subheader('Variables anonymisées (V1-V28)')
    V_inputs = []
    for i in range(1, 29):
        V_inputs.append(st.sidebar.number_input(f'V{i}', -50.0, 50.0, 0.0, 0.01))

    # Créer un dictionnaire pour le DataFrame d'entrée
    user_data = {
        'Time': Time,
        'Amount': Amount,
    }
    for i in range(1, 29):
        user_data[f'V{i}'] = V_inputs[i - 1]

    features = pd.DataFrame(user_data, index=[0])
    return features


# Obtenir les entrées utilisateur
input_df = get_user_input()

st.subheader('Caractéristiques de la Transaction Soumise')
st.write(input_df)

# Prédiction
if st.button('Prédire la Fraude'):
    # L'ordre des colonnes doit correspondre à celui utilisé pendant l'entraînement
    # En raison de la manière dont les colonnes V1-V28 sont nommées, nous devons les trier si elles ne sont pas dans le bon ordre
    # Cependant, StandardScaler attend les mêmes colonnes et dans le même ordre que lors de fit.
    # Notre pipeline gère l'ordre des colonnes V1-V28 automatiquement car elles sont des features.
    # Nous devons nous assurer que Time et Amount sont aux bonnes positions si elles ne sont pas déjà.

    # Récupérer l'ordre des colonnes d'entraînement depuis le pipeline (scaler)
    # Note: Le scaler étant le premier pas du pipeline, il a gardé les colonnes.
    # Si l'attribut feature_names_in_ existe (Scikit-learn 1.0+),
    # nous pouvons l'utiliser pour garantir l'ordre.

    # Pour simplifier et assurer la compatibilité, nous allons reconstruire le DF d'entrée
    # avec les colonnes dans l'ordre attendu par le modèle.
    # L'ordre des colonnes dans le X_train initial était : Time, V1, V2, ..., V28, Amount.

    # Créer une liste ordonnée des noms de colonnes attendus par le modèle
    # La variable X contient l'ordre des colonnes utilisées pour l'entraînement
    expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    # S'assurer que le DataFrame d'entrée a les mêmes colonnes dans le même ordre
    input_ordered_df = input_df[expected_columns]

    prediction = model_pipeline.predict(input_ordered_df)
    prediction_proba = model_pipeline.predict_proba(input_ordered_df)[:, 1]

    st.subheader('Résultat de la Prédiction')
    if prediction[0] == 1:
        st.error(f"**Transaction Frauduleuse !** Probabilité : {prediction_proba[0]:.2f}")
    else:
        st.success(f"Transaction Normale. Probabilité de fraude : {prediction_proba[0]:.2f}")

    st.write("Note: Une probabilité élevée ne garantit pas la fraude, mais indique un risque.")