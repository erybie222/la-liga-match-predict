import csv
from datetime import datetime
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
history_path = os.path.join(os.path.dirname(__file__), '..', 'history', 'predictions.csv')
if os.path.exists(history_path):
    df_history = pd.read_csv(history_path)
    st.dataframe(df_history.tail(5).iloc[::-1],use_container_width=True)
else:
    st.info("No predictions yet.")
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

            history_path = os.path.join(os.path.dirname(__file__), '..', 'history', 'predictions.csv')
            os.makedirs(os.path.dirname(history_path), exist_ok=True)

            with open(history_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if os.stat(history_path).st_size == 0:
                    writer.writerow(["timestamp", "home_team", "away_team", "prediction", "proba_H", "proba_D", "proba_A"])

                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    home,
                    away,
                    pred_label,
                    round(proba[0], 3),
                    round(proba[1], 3),
                    round(proba[2], 3)
                ])
        except Exception as e:
            st.error(f"Error: {str(e)}")