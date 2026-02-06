import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
import os

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to project root
ROOT_DIR = os.path.dirname(BASE_DIR)

# Construct model paths safely
model_path = os.path.join(ROOT_DIR, "model", "churn_model.pkl")
features_path = os.path.join(ROOT_DIR, "model", "feature_columns.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)


st.title("📉 Customer Churn Prediction System")

st.markdown("Enter customer details below to predict churn risk.")

# --- INPUTS ---
st.set_page_config(page_title="Churn Prediction System", layout="wide")

st.title("📉 Customer Churn Prediction System")
st.markdown("Predict customer churn risk using ML model.")

st.sidebar.header("Customer Inputs")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Create input dataframe
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "PaymentMethod": payment_method
}

input_df = pd.DataFrame([input_dict])

# One-hot encode
input_df = pd.get_dummies(input_df)

# Align with training features
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{probability:.2%}")

    if probability < 0.30:
        st.success("🟢 Low Risk Customer")
    elif probability < 0.60:
        st.warning("🟡 Moderate Risk Customer")
    else:
        st.error("🔴 High Churn Risk")

    st.progress(int(probability * 100))


