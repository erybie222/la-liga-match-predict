import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from utils.preprocessing import prepare_dataset
from imblearn.over_sampling import SMOTE



def tune_xgboost():
    X, y , le_ftr= prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    le_ftr = joblib.load('models/label_encoder_smote.pkl')
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )

    param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 1.5, 2],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1]  
    }


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1_macro',
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Najlepsze parametry:", search.best_params_)
    print("Najlepszy F1_macro (CV):", search.best_score_)

    print("\n=== Raport końcowy ===")
    print(classification_report(y_test, y_pred, target_names=le_ftr.classes_))

    importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Feature importance ===")
    print(importance)

    importance.plot(kind='bar', title='XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()

    joblib.dump(best_model, 'models/best_model_tuned_smote.pkl')

    df_results = pd.DataFrame(search.cv_results_)
    df_results.to_csv('results/tuning_results.csv', index=False)


    return best_model


if __name__ == '__main__':
    tune_xgboost()
