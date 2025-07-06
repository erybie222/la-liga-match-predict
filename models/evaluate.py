import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import streamlit as st

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_

    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Greens"), use_container_width=True)
