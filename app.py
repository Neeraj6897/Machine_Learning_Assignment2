import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

st.title("Bank Marketing: ML Classification Models")
st.write("This app demonstrates 5 classification models trained on the Bank Marketing dataset.")
st.write("Upload the test data CSV file and select a model to see the evaluation results.")

#load saved models
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree': joblib.load('model/decision_tree.pkl'),
        'KNN': joblib.load('model/knn.pkl'),
        'Naive Bayes': joblib.load('model/naive_bayes.pkl'),
        'Random Forest (Ensemble)': joblib.load('model/random_forest.pkl')
    }
    scaler = joblib.load('model/scaler.pkl')
    return models, scaler

models, scaler = load_models()

#file upload section
st.subheader("Upload Test Data")
uploaded_file = st.file_uploader("Upload your test data CSV file", type=['csv'])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(test_data.head())

    #check if target column exists
    if 'y' not in test_data.columns:
        st.error("The uploaded CSV must contain a 'y' column (target variable).")
    else:
        X_test = test_data.drop('y', axis=1)
        y_test = test_data['y']

        #scale the features
        X_test_scaled = scaler.transform(X_test)

        #model selection dropdown
        st.subheader("Select a Model")
        model_name = st.selectbox("Choose a classification model:",
                                  list(models.keys()))

        selected_model = models[model_name]

        #run predictions
        y_pred = selected_model.predict(X_test_scaled)
        y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]

        #calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        #display evaluation metrics
        st.subheader(f"Evaluation Metrics - {model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        #confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
        st.pyplot(fig)

        #classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)

        #comparison of all models
        st.subheader("All Models Comparison")
        all_results = []
        for name, model in models.items():
            y_p = model.predict(X_test_scaled)
            y_pr = model.predict_proba(X_test_scaled)[:, 1]
            all_results.append({
                'Model': name,
                'Accuracy': round(accuracy_score(y_test, y_p), 4),
                'AUC': round(roc_auc_score(y_test, y_pr), 4),
                'Precision': round(precision_score(y_test, y_p), 4),
                'Recall': round(recall_score(y_test, y_p), 4),
                'F1': round(f1_score(y_test, y_p), 4),
                'MCC': round(matthews_corrcoef(y_test, y_p), 4)
            })

        results_df = pd.DataFrame(all_results)
        st.dataframe(results_df.set_index('Model'))

else:
    st.info("Please upload the test_data.csv file to get started.")
