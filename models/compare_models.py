import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from utils.preprocessing import get_preprocessed_data

# Wczytaj dane testowe
X, y, X_train, X_test, y_train, y_test, le_ftr = get_preprocessed_data()

# Modele do porównania
model_paths = {
    "No SMOTE": "models/best_model.pkl",
    "SMOTE": "models/best_model_smote.pkl",
    "Tuned SMOTE": "models/best_model_tuned_smote.pkl"
}

# Wyniki
results = []

# Porównanie
for name, path in model_paths.items():
    print(f"\n🔍 Model: {name}")
    model = joblib.load(path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le_ftr.classes_, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Macro": f1_macro
    })

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le_ftr.classes_))

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_ftr.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.show()

# Tabela porównawcza
df = pd.DataFrame(results).set_index("Model")
print("\n📊 Podsumowanie:")
print(df)

df.plot(kind='barh', title='Porównanie modeli: Accuracy vs F1 Macro')
plt.xlabel("Wartość")
plt.tight_layout()
plt.show()
