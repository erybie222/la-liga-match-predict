from fastapi import FastAPI, Query
import pandas as pd
from pydantic import BaseModel
import joblib
from utils.preprocessing import prepare_match_features
import numpy as np

app = FastAPI()

model = joblib.load('models/best_model_tuned_smote.pkl')
label_encoder = joblib.load('models/label_encoder_smote.pkl')

from typing import Union, Dict

class PredictionResponse(BaseModel):
    prediction: Union[str, None] = None
    probabilities: Union[Dict[str, float], None] = None
    error: Union[str, None] = None

class MatchRequest(BaseModel):
    home:str
    away:str

@app.get("/predict", response_model=PredictionResponse)
def predict(home: str = Query(...), away: str = Query(...)):
    try:
        features = prepare_match_features(home, away)

        if features.empty:
            return {"error": "Nie udało się przygotować cech – brak danych."}

        y_pred = model.predict(features)[0]
        y_proba = model.predict_proba(features)[0]

        prediction_label = label_encoder.inverse_transform([y_pred])[0]
        class_probs = {
            label_encoder.inverse_transform([i])[0]: round(float(prob), 3)
            for i, prob in enumerate(y_proba)
        }

        return {"prediction": prediction_label, "probabilities": class_probs}

    except Exception as e:
        print("Błąd przy predykcji:", str(e))
        return {"error": str(e)}


@app.get("/teams")
def list_teams():
    df = pd.read_csv('data/LaLiga_Matches.csv')
    teams = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
    return {"teams": teams}

@app.post("/predcit_json", response_model=PredictionResponse)
def predict_json(request: MatchRequest):
    return predict(request.home, request.away)