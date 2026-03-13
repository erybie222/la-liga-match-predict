# la-liga-match-predict

Simple LaLiga match prediction project using ELO and other match features.

Requirements

- Python 3.10+
- Install: `python -m pip install -r requirements.txt`

Quick start

- Run API: `uvicorn api.main:app --reload` (GET /predict?home=TeamA&away=TeamB)
- Run dashboard: `streamlit run dashboard/app.py`

Notes

- Models are in `models/` (e.g. `best_model_tuned_smote.pkl`).
- Use the matching `label_encoder` and same preprocessing when predicting.
- If classes are imbalanced, prefer SMOTE-trained models.
