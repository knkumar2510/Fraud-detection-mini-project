import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    layout="wide"
)

st.title("💳 Online Payment Fraud Detection System")
st.markdown("Machine Learning based fraud detection using transactional data")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("D:\Fraud Detection/Credit_Card_Applications.csv")

df = load_data()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Exploratory Data Analysis", "Model Training", "Fraud Prediction"]
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if menu == "Overview":
    st.header("📌 Project Overview")

    st.markdown("""
    **Objective:**  
    Detect fraudulent online transactions using machine learning.

    **Dataset:**  
    Online Payment Fraud Dataset  

    **Target Variable:**  
    - `isFraud` (1 = Fraud, 0 = Legit)

    **Techniques Used:**  
    - SMOTE for class imbalance  
    - Feature scaling & encoding  
    - Multiple ML models  
    - Hyperparameter tuning  
    """)

    col1, col2 = st.columns(2)
    col1.metric("Total Transactions", df.shape[0])
    col2.metric("Fraud Cases", int(df["isFraud"].sum()))

    st.dataframe(df.head())

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif menu == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Fraud Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="isFraud", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["amount"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Transaction Type vs Fraud")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x="type", hue="isFraud", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
elif menu == "Model Training":
    st.header("🤖 Model Training & Evaluation")

    categorical_cols = ["type"]
    numeric_cols = [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]
    target = "isFraud"

    X = df[categorical_cols + numeric_cols]
    y = df[target]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1),
        "KNN": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    results = {}

    with st.spinner("Training models..."):
        for name, model in models.items():
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_test)

            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred)
            }

    results_df = pd.DataFrame(results).T.sort_values(by="F1", ascending=False)
    st.subheader("📈 Model Comparison")
    st.dataframe(results_df)

    # Save best model
    best_model_name = results_df.index[0]
    best_model = models[best_model_name]

    joblib.dump(best_model, "model/best_model.pkl")
    joblib.dump(preprocessor, "model/preprocessor.pkl")

    st.success(f"✅ Best Model Saved: {best_model_name}")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
elif menu == "Fraud Prediction":
    st.header("🔍 Real-Time Fraud Prediction")

    model = joblib.load("model/best_model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input("Step", 0)
        amount = st.number_input("Amount", 0.0)
        oldbalanceOrg = st.number_input("Old Balance Origin", 0.0)
        newbalanceOrig = st.number_input("New Balance Origin", 0.0)

    with col2:
        oldbalanceDest = st.number_input("Old Balance Destination", 0.0)
        newbalanceDest = st.number_input("New Balance Destination", 0.0)
        tx_type = st.selectbox("Transaction Type", df["type"].unique())

    if st.button("Predict Fraud"):
        input_df = pd.DataFrame([{
            "type": tx_type,
            "step": step,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        processed = preprocessor.transform(input_df)
        prediction = model.predict(processed)[0]

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Transaction is Legitimate")

