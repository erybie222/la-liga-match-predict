# models/tune_model.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import prepare_dataset  # <- refaktoryzowana funkcja do ładowania danych

def tune_random_forest():
    # Przygotowanie danych
    X, y, label_encoder = prepare_dataset()  # zakładamy, że funkcja zwraca X, y, le_ftr
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Parametry
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Wyniki
    print("Najlepsze parametry:", grid_search.best_params_)
    print("Najlepszy F1_macro (CV):", grid_search.best_score_)

    print("\n=== Raport końcowy ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Feature importance
    importances = best_model.feature_importances_
    feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("\n=== Feature importance ===")
    print(feature_importance)

    feature_importance.plot(kind='bar', title='Feature Importance')
    plt.tight_layout()
    plt.show()

    return best_model  # <- do ew. zapisu lub dalszej ewaluacji


# Główne wywołanie
if __name__ == '__main__':
    tune_random_forest()
