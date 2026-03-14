import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("deepcsat_xgboost_model.joblib")

st.title("DeepCSAT - Ecommerce Customer Satisfaction Prediction")

st.write("Predict Customer Satisfaction Score using Machine Learning")

# Inputs
connected_handling_time = st.number_input("Handling Time")
response_time_minutes = st.number_input("Response Time (minutes)")
issue_hour = st.number_input("Issue Hour")
item_price = st.number_input("Item Price")
remark_length = st.number_input("Customer Remark Length")

# convert to array
input_data = np.array([[connected_handling_time,
                        response_time_minutes,
                        issue_hour,
                        item_price,
                        remark_length]])

if st.button("Predict CSAT Score"):
    
    prediction = model.predict(input_data)
    
    # convert back from 0–4 to 1–5
    prediction = prediction + 1
    
    st.success(f"Predicted Customer Satisfaction Score: {prediction[0]}")