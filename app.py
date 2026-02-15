import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="AI Credit Risk Engine", layout="wide")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("credit_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

# -------------------------------
# Premium Header
# -------------------------------
st.markdown("""
<h1 style='text-align:center; color:#1f4e79;'>AI Powered Credit Risk Engine</h1>
<h4 style='text-align:center;'>Explainable AI | India-focused Alternative Scoring</h4>
<hr>
""", unsafe_allow_html=True)

# -------------------------------
# Applicant Type Selection
# -------------------------------
applicant_type = st.radio("Select Applicant Type", ["Individual", "MSME"], horizontal=True)

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("Applicant Details")

input_dict = {}

if applicant_type == "Individual":

    col1, col2, col3 = st.columns(3)

    with col1:
        input_dict["age"] = st.number_input("Age", 18, 70, 30)
        input_dict["monthly_income"] = st.number_input("Monthly Income", 10000, 500000, 50000)
        input_dict["employment_years"] = st.number_input("Employment Years", 0, 40, 5)
        input_dict["credit_score"] = st.number_input("Credit Score", 300, 900, 650)

    with col2:
        input_dict["savings_balance"] = st.number_input("Savings Balance", 0, 1000000, 50000)
        input_dict["existing_loans"] = st.number_input("Existing Loans", 0, 10, 1)
        input_dict["loan_amount"] = st.number_input("Loan Amount", 10000, 2000000, 200000)
        input_dict["loan_duration"] = st.number_input("Loan Duration (Months)", 6, 120, 24)

    with col3:
        input_dict["emi_ratio"] = st.slider("EMI Ratio", 0.0, 1.0, 0.3)
        input_dict["digital_payment_frequency"] = st.number_input("Digital Payments / Month", 0, 500, 50)
        input_dict["utility_bill_payment_score"] = st.slider("Utility Bill Payment Score", 0, 100, 80)
        input_dict["dependents"] = st.number_input("Dependents", 0, 10, 2)

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        input_dict["business_age"] = st.number_input("Business Age (Years)", 0, 30, 5)
        input_dict["annual_turnover"] = st.number_input("Annual Turnover", 100000, 100000000, 2000000)
        input_dict["gst_filing_score"] = st.slider("GST Filing Score", 0, 100, 75)

    with col2:
        input_dict["average_monthly_cashflow"] = st.number_input("Avg Monthly Cashflow", 10000, 5000000, 200000)
        input_dict["existing_business_loans"] = st.number_input("Existing Business Loans", 0, 10, 1)
        input_dict["sector_risk_score"] = st.slider("Sector Risk Score", 0.0, 1.0, 0.5)

    with col3:
        input_dict["supplier_payment_score"] = st.slider("Supplier Payment Score", 0, 100, 80)
        input_dict["upi_transaction_volume"] = st.number_input("UPI Transactions / Month", 0, 5000, 300)
        input_dict["inventory_turnover_ratio"] = st.slider("Inventory Turnover Ratio", 0.1, 15.0, 3.0)
        input_dict["profit_margin"] = st.slider("Profit Margin", 0.0, 0.5, 0.15)
        input_dict["credit_history_score"] = st.slider("Credit History Score", 0, 100, 70)
        input_dict["loan_amount"] = st.number_input("Loan Amount", 10000, 2000000, 500000)
        input_dict["loan_duration"] = st.number_input("Loan Duration (Months)", 6, 120, 36)

# Fill missing features with 0
for feature in feature_names:
    if feature not in input_dict:
        input_dict[feature] = 0

# Create DataFrame
input_data = pd.DataFrame([input_dict])[feature_names]

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Analyze Credit Risk"):

    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success(f"Approved ✅ | Default Risk: {prob:.2%}")
    else:
        st.error(f"High Risk ❌ | Default Risk: {prob:.2%}")

    # ----------------------------------
    # ROC Curve
    # ----------------------------------
    st.subheader("ROC Curve")

    y_prob = model.predict_proba(input_data)[:,1]
    fpr, tpr, _ = roc_curve([0], y_prob)
    roc_auc = auc(fpr, tpr)

    fig1 = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.pyplot(fig1)

    # ----------------------------------
    # Confusion Matrix (Simulated)
    # ----------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix([0], model.predict(input_data))

    fig2 = plt.figure()
    plt.imshow(cm)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig2)

    # ----------------------------------
    # SHAP Explainability
    # ----------------------------------
    st.subheader("Explainable AI (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig3 = plt.figure()
    shap.summary_plot(shap_values, input_data, show=False)
    st.pyplot(fig3)
