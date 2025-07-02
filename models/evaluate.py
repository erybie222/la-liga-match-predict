import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Wczytanie danych i label encodera
from utils.preprocessing import get_preprocessed_data
X, y, X_train, X_test, y_train, y_test, le_ftr = get_preprocessed_data()

# Wczytanie modelu
model = joblib.load('models/best_model.pkl')

# Predykcja
y_pred = model.predict(X_test)


# 1. Dokładność
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Dokładność modelu: {accuracy:.2%}")

# 2. Raport klasyfikacji
print("\n📋 Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=le_ftr.classes_))

# 3. Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_ftr.classes_)
disp.plot(cmap='Blues')
plt.title("🔍 Macierz pomyłek")
plt.tight_layout()
plt.show()
