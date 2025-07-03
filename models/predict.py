import joblib

def predict():
    model = joblib.load('models/best_model_tuned_smote.pkl')

    home_team = input('Enter home team name: ')
    away_team = input('Enter away team name: ')