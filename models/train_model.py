import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report
from utils.preprocessing import get_preprocessed_data

X, y, X_train, X_test, y_train, y_test, le_ftr = get_preprocessed_data()

model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(le_ftr.classes_),
    n_estimators=100,
    max_depth=10,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le_ftr.classes_))

joblib.dump(model, 'models/best_model.pkl')
joblib.dump(le_ftr, 'models/label_encoder.pkl')

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n📊 Feature importance:")
print(importance)

importance.plot(kind='bar', title='XGBoost Feature Importance')
plt.tight_layout()
plt.show()
