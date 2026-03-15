import streamlit as st
import joblib
import pandas as pd
import numpy as np
# -----------------------------
# Load Saved Objects
# -----------------------------

@st.cache_resource
def load_model():
    model = joblib.load("deepcsat_model.joblib")
    scaler = joblib.load("scaler.joblib")
    imputer = joblib.load("imputer.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    return model, scaler, imputer, feature_columns

model, scaler, imputer, feature_columns = load_model()
# model = joblib.load("deepcsat_model.joblib")
# scaler = joblib.load("scaler.joblib")
# imputer = joblib.load("imputer.joblib")
# feature_columns = joblib.load("feature_columns.joblib")

# -----------------------------
# App Title
# -----------------------------
st.title("DeepCSAT – Ecommerce Customer Satisfaction Prediction")
st.write("Predict Customer Satisfaction Score using Machine Learning")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Customer Support Details")

connected_handling_time = st.number_input("Connected Handling Time", min_value=0.0)
response_time_minutes = st.number_input("Response Time (minutes)", min_value=0.0)
order_to_issue_hours = st.number_input("Order to Issue Hours", min_value=0.0)
survey_delay_hours = st.number_input("Survey Delay Hours", min_value=0.0)

item_price = st.number_input("Item Price", min_value=0.0)
remark_length = st.number_input("Customer Remark Length", min_value=0)

issue_hour = st.slider("Issue Hour", 0, 23)
order_hour = st.slider("Order Hour", 0, 23)

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict CSAT Score"):

    # Create input dictionary
    input_dict = {
        "connected_handling_time": connected_handling_time,
        "response_time_minutes": response_time_minutes,
        "order_to_issue_hours": order_to_issue_hours,
        "survey_delay_hours": survey_delay_hours,
        "Item_price": item_price,
        "remark_length": remark_length,
        "issue_hour": issue_hour,
        "order_hour": order_hour
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Add missing columns with 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns exactly like training
    input_df = input_df[feature_columns]

    # Apply preprocessing
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = model.predict(input_scaled)

    # Convert back from 0–4 to 1–5
    prediction = prediction + 1

    st.success(f"Predicted Customer Satisfaction Score: {prediction[0]}")

