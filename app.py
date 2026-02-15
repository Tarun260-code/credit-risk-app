import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Credit Engine",
    layout="wide",
)

# ------------------------------------------------
# CLEAN PREMIUM CSS
# ------------------------------------------------
st.markdown("""
<style>

/* Remove Streamlit default padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Page background */
body {
    background-color: #0b1220;
}

/* Gradient Header */
.header {
    background: linear-gradient(90deg, #0f172a, #1e293b);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
}

/* Card */
.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    margin-bottom: 25px;
    border: 1px solid #1f2937;
}

/* Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #e5e7eb;
}

/* Metric styling */
div[data-testid="metric-container"] {
    background-color: #0f172a;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    border: none;
}

/* Approval Badges */
.approve { color: #22c55e; font-size: 24px; font-weight: 600; }
.conditional { color: #facc15; font-size: 24px; font-weight: 600; }
.reject { color: #ef4444; font-size: 24px; font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown("""
<div class="header">
    <h1>🏦 AI Powered Credit Risk Engine</h1>
    <p>Explainable AI for Indian Individuals & MSMEs</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("credit_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, feature_names = load_model()

# ------------------------------------------------
# INPUT CARD
# ------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Applicant Details</div>', unsafe_allow_html=True)

applicant_type = st.radio("Applicant Type", ["Individual", "MSME"])

loan_amount = st.number_input("Loan Amount (₹)", min_value=1000)
loan_duration = st.number_input("Loan Duration (Months)", min_value=1)
existing_emis = st.number_input("Existing EMIs (₹)", min_value=0)
missed_payments = st.number_input("Missed Payments", min_value=0)
cheque_bounce = st.number_input("Cheque Bounce Count", min_value=0)

if applicant_type == "Individual":
    age = st.number_input("Age", min_value=18)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0)
    work_experience = st.number_input("Work Experience (Years)", min_value=0)
    savings = st.number_input("Savings (₹)", min_value=0)
else:
    business_age = st.number_input("Business Age (Years)", min_value=0)
    monthly_revenue = st.number_input("Monthly Revenue (₹)", min_value=0)
    annual_turnover = st.number_input("Annual Turnover (₹)", min_value=0)
    gst_registered = st.selectbox("GST Registered?", [0, 1])
    employee_count = st.number_input("Employee Count", min_value=0)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
if st.button("Evaluate Credit Risk"):

    input_dict = {
        "loan_amount": loan_amount,
        "loan_duration": loan_duration,
        "existing_emis": existing_emis,
        "missed_payments": missed_payments,
        "cheque_bounce": cheque_bounce,
    }

    if applicant_type == "Individual":
        input_dict.update({
            "age": age,
            "monthly_income": monthly_income,
            "work_experience": work_experience,
            "savings": savings,
        })
    else:
        input_dict.update({
            "business_age": business_age,
            "monthly_revenue": monthly_revenue,
            "annual_turnover": annual_turnover,
            "gst_registered": gst_registered,
            "employee_count": employee_count,
        })

    for f in feature_names:
        if f not in input_dict:
            input_dict[f] = 0

    input_df = pd.DataFrame([input_dict])[feature_names]

    prob = model.predict_proba(input_df)[0][1]
    risk_score = round(prob * 100, 2)

    # ---------------------------------------------
    # RISK CARD
    # ---------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Default Probability (%)", risk_score)

    if risk_score < 30:
        category = "Low Risk"
        grade = "A"
    elif risk_score < 60:
        category = "Medium Risk"
        grade = "B"
    else:
        category = "High Risk"
        grade = "C"

    col2.metric("Risk Grade", grade)
    st.write(f"**Category:** {category}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------
    # APPROVAL CARD
    # ---------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Approval Recommendation</div>', unsafe_allow_html=True)

    if applicant_type == "Individual":
        dti = (existing_emis + (loan_amount / loan_duration)) / max(monthly_income,1)
        st.metric("Debt-to-Income Ratio", round(dti,2))

        if risk_score < 30 and dti < 0.4:
            st.markdown('<p class="approve">✅ APPROVED</p>', unsafe_allow_html=True)
        elif risk_score < 60 and dti < 0.6:
            st.markdown('<p class="conditional">🟡 CONDITIONAL APPROVAL</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="reject">❌ REJECTED</p>', unsafe_allow_html=True)
    else:
        coverage = monthly_revenue / max((loan_amount/loan_duration),1)
        st.metric("Revenue Coverage Ratio", round(coverage,2))

        if risk_score < 30 and coverage > 2:
            st.markdown('<p class="approve">✅ APPROVED</p>', unsafe_allow_html=True)
        elif risk_score < 60 and coverage > 1.2:
            st.markdown('<p class="conditional">🟡 CONDITIONAL APPROVAL</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="reject">❌ REJECTED</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------
    # SHAP CARD
    # ---------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explainable AI</div>', unsafe_allow_html=True)

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
