import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
from utils.preprocessing import prepare_match_features 
from sklearn.exceptions import NotFittedError

def predict():
    model_path = 'models/best_model_tuned_smote.pkl'
    encoder_path = 'models/label_encoder_smote.pkl'

    try:
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
    except FileNotFoundError as e:
        print(f"❌ Błąd: {e}")
        return

    home_team = input("🏠 Podaj nazwę drużyny gospodarzy: ").strip()
    away_team = input("🚗 Podaj nazwę drużyny gości: ").strip()

    try:
        X = prepare_match_features(home_team, away_team)
    except Exception as e:
        print(f"❌ Błąd przy przygotowaniu cech: {e}")
        return

    try:
        y_pred = model.predict(X)
        result = le.inverse_transform(y_pred)[0]
    except NotFittedError:
        print("❌ Model nie został wytrenowany.")
        return

    outcome = {
        'H': f"🔵 {home_team} wygra",
        'A': f"🔴 {away_team} wygra",
        'D': "🟡 Remis"
    }

    print(f"\n📊 Przewidywany wynik meczu: {outcome.get(result, 'Nieznany wynik')}")

if __name__ == '__main__':
    predict()
