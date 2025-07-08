import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
import joblib

def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    tuned_model = joblib.load(os.path.join(os.path.dirname(__file__), 'best_model_tuned_smote.pkl'))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    }

    results = []

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average='weighted'),
            "Log Loss": log_loss(y_test, y_proba)
        })

    y_pred = tuned_model.predict(X_test)
    y_proba = tuned_model.predict_proba(X_test)
    results.append({
        "Model": "XGBoost tuned with SMOTE",
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "Log Loss": log_loss(y_test, y_proba)
    })

    df_results = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    return df_results
