import shap
import pandas as pd

def get_shap_explanation(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values, explainer
