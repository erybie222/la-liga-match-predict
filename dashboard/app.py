import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import joblib
from utils.preprocessing import prepare_match_features

model = joblib.load('../models/best_model_tuned_smote.pkl')
label_encoder = joblib.load('../models/label_encoder_smote.pkl')

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'LaLiga_Matches.csv'), parse_dates=['Date'])
teams = sorted(set(df['HomeTeam']) | set(df["AwayTeam"]))

st.title("⚽ La Liga Match Predictor")
home = st.selectbox("🏠 Choose home team", teams)
away = st.selectbox("🚗 Choose away team", teams)

if home == away:
    st.warning("Teams must be different!")
else:
    if st.button("🔮 Predict result"):
        try:
            features = prepare_match_features(home, away)
            st.subheader("📋 Input features: ")
            st.write(features.T)

            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            pred_label = label_encoder.inverse_transform([pred])[0]
            
            st.success(f"**Predicted result: {pred_label}**")

            st.subheader("📊 Probabilities:")
            for i, p in enumerate(proba):
                label = label_encoder.inverse_transform([i])[0]
                st.write(f"{label}: {round(p*100, 2)}%")

        except Exception as e:
            st.error(f"Error: {str(e)}")