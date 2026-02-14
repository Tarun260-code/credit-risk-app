import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature order
model = joblib.load("credit_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Credit Risk Prediction App")
st.write("Fill in the details below to assess the probability of loan default.")

# -----------------------
# User Inputs (Clean Hackathon UI)
# -----------------------

age = st.number_input("Your Age", min_value=18, max_value=100, value=30)

credit_amount = st.number_input(
    "Loan Amount Requested",
    min_value=1000,
    value=50000
)

duration = st.number_input(
    "Loan Duration (in months)",
    min_value=1,
    value=12
)

employment = st.number_input(
    "How many years have you worked?",
    min_value=0,
    value=3
)

savings = st.number_input(
    "Your Savings",
    min_value=0,
    value=10000
)

missed_payments = st.selectbox(
    "Have you missed payments before?",
    ["No", "Yes"]
)

# Convert to numeric for model
payment_status = 1 if missed_payments == "Yes" else 0

# -----------------------
# Prediction Button
# -----------------------

if st.button("Check Credit Risk"):

    # Hidden default values for remaining features
    input_dict = {
        'Account Balance': 50000,
        'Duration of Credit (month)': duration,
        'Payment Status of Previous Credit': payment_status,
        'Purpose': 1,
        'Credit Amount': credit_amount,
        'Value Savings/Stocks': savings,
        'Length of current employment': employment,
        'Instalment per cent': 2,
        'Sex & Marital Status': 1,
        'Guarantors': 0,
        'Duration in Current address': 2,
        'Most valuable available asset': 2,
        'Age (years)': age,
        'Concurrent Credits': 1,
        'Type of apartment': 1,
        'No of Credits at this Bank': 1,
        'Occupation': 2,
        'No of dependents': 1,
        'Telephone': 1,
        'Foreign Worker': 1
    }

    # Create dataframe in correct order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict_proba(input_df)[:, 1]
    risk_percent = float(round(prediction[0] * 100, 2))

    # Display result
    st.subheader(f"Predicted Default Risk: {risk_percent}%")

    # Progress bar (visual risk indicator)
    st.progress(risk_percent / 100)

    # Final decision message
    if risk_percent > 50:
        st.error("⚠ High Credit Risk – Loan Approval Not Recommended")
    else:
        st.success("✅ Low Credit Risk – Loan Approval Recommended")
