
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.preprocessing import get_preprocessed_data

# Wczytanie danych
X, y, X_train, X_test, y_train, y_test, le_ftr = get_preprocessed_data()

# Trening modelu Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# Raport klasyfikacji
print(classification_report(y_test, y_pred, target_names=le_ftr.classes_))

# Zapis modelu i label encodera
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(le_ftr, 'models/label_encoder.pkl')

# (Opcjonalnie) Ważność cech
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Wydruk i wykres
print("\n📊 Feature importance:")
print(feature_importance)

feature_importance.plot(kind='bar', title='Feature Importance')
plt.tight_layout()
plt.show()
