import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

matplotlib.use("Agg")

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="AI Credit Risk System", layout="centered")
st.title("AI Credit Risk Evaluation")

st.write("Enter applicant details manually:")

# ======================
# MANUAL INPUTS
# ======================

CreditAmount = st.number_input("Credit Amount (₹)", value=5000)

Duration = st.number_input("Loan Duration (months)", value=24)

Age = st.number_input("Age (years)", value=30)

MonthlySalary = st.number_input("Monthly Salary (₹)", value=25000)

SavingsAmount = st.number_input("Savings Amount (₹)", value=10000)

Employment_ui = st.selectbox(
    "Employment Status",
    ["Unemployed", "Less than 1 year", "1 to 4 years", "4 to 7 years", "More than 7 years"]
)

PaymentStatus_ui = st.selectbox(
    "Past Loan Repayment",
    ["No previous loans", "All loans paid on time", "Existing loans being paid", "Some delays in payment", "Critical / defaulted before"]
)

Housing_ui = st.selectbox(
    "Type of Housing",
    ["Rented", "Owned", "Free / Provided by employer"]
)

# ======================
# MAPPING
# ======================

def map_employment(val):
    if val == "Unemployed":          return 1
    if val == "Less than 1 year":    return 2
    if val == "1 to 4 years":        return 3
    if val == "4 to 7 years":        return 4
    return 5

def map_housing(val):
    if val == "Rented": return 1
    if val == "Owned":  return 2
    return 3

def map_payment(val):
    if val == "No previous loans":              return 0
    if val == "All loans paid on time":         return 1
    if val == "Existing loans being paid":      return 2
    if val == "Some delays in payment":         return 3
    return 4

def salary_to_instalment(salary, credit, duration):
    if salary <= 0:
        return 2
    emi = credit / duration if duration > 0 else credit
    ratio = (emi / salary) * 100
    if ratio < 10:  return 1
    if ratio < 20:  return 2
    if ratio < 35:  return 3
    return 4

def savings_to_code(savings):
    if savings <= 0:      return 5
    if savings < 10000:   return 1
    if savings < 50000:   return 2
    if savings < 100000:  return 3
    return 4

# ======================
# PREDICTION
# ======================

if st.button("Evaluate Credit Risk"):

    # Columns in exact same order as training data
    input_df = pd.DataFrame([[
        2,                                                          # Account Balance
        Duration,                                                   # Duration of Credit (month)
        map_payment(PaymentStatus_ui),                              # Payment Status of Previous Credit
        3,                                                          # Purpose
        CreditAmount,                                               # Credit Amount
        savings_to_code(SavingsAmount),                             # Value Savings/Stocks
        map_employment(Employment_ui),                              # Length of current employment
        salary_to_instalment(MonthlySalary, CreditAmount, Duration),# Instalment per cent
        3,                                                          # Sex & Marital Status
        1,                                                          # Guarantors
        2,                                                          # Duration in Current address
        2,                                                          # Most valuable available asset
        Age,                                                        # Age (years)
        3,                                                          # Concurrent Credits
        map_housing(Housing_ui),                                    # Type of apartment
        1,                                                          # No of Credits at this Bank
        3,                                                          # Occupation
        1,                                                          # No of dependents
        1,                                                          # Telephone
        1,                                                          # Foreign Worker
    ]], columns=[
        "Account Balance", "Duration of Credit (month)",
        "Payment Status of Previous Credit", "Purpose", "Credit Amount",
        "Value Savings/Stocks", "Length of current employment",
        "Instalment per cent", "Sex & Marital Status", "Guarantors",
        "Duration in Current address", "Most valuable available asset",
        "Age (years)", "Concurrent Credits", "Type of apartment",
        "No of Credits at this Bank", "Occupation", "No of dependents",
        "Telephone", "Foreign Worker",
    ])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ── Result ──────────────────────────────
    st.subheader("Result")

    if prediction == 1:
        st.success("Low Credit Risk ✅")
    else:
        st.error("High Credit Risk ⚠")

    st.write(f"Default Probability: {round(probability, 4)}")

    st.divider()

    # ── SHAP Waterfall ───────────────────────
    st.subheader("Why this prediction? (SHAP)")

    try:
        # Get the final estimator from pipeline if model is a pipeline
        if hasattr(model, "named_steps"):
            # It's a pipeline — transform input then explain final step
            preprocessed = model[:-1].transform(input_df)
            final_model   = model[-1]
            feature_names = input_df.columns.tolist()

            explainer   = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(preprocessed)

            if isinstance(shap_values, list):
                sv     = shap_values[1][0]
                base_v = explainer.expected_value[1]
            else:
                sv     = shap_values[0]
                base_v = float(explainer.expected_value)

            data_row = preprocessed[0]
        else:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                sv     = shap_values[1][0]
                base_v = explainer.expected_value[1]
            else:
                sv     = shap_values[0]
                base_v = float(explainer.expected_value)

            data_row      = input_df.iloc[0].values
            feature_names = input_df.columns.tolist()

        explanation = shap.Explanation(
            values        = sv,
            base_values   = base_v,
            data          = data_row,
            feature_names = feature_names,
        )

        fig, ax = plt.subplots()
        shap.waterfall_plot(explanation, show=False)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning(f"SHAP could not run: {e}")

    st.divider()

    # ── Load real data for ROC and Confusion Matrix ──
    try:
        from sklearn.model_selection import train_test_split

        df = pd.read_csv("german_credit.csv")
        X  = df.drop(columns=["Creditability"])
        y  = df["Creditability"]

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = model.predict(X_test)

        data_loaded = True
    except Exception as e:
        data_loaded = False
        st.warning(f"Could not load german_credit.csv: {e}")

    # ── ROC Curve ────────────────────────────
    st.subheader("ROC Curve")

    if data_loaded:
        try:
            fpr, tpr, _ = roc_curve(y_test, test_probs)
            auc         = roc_auc_score(y_test, test_probs)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr, color="steelblue", label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.warning(f"ROC curve could not be drawn: {e}")

    st.divider()

    # ── Confusion Matrix ─────────────────────
    st.subheader("Confusion Matrix")

    if data_loaded:
        try:
            cm = confusion_matrix(y_test, test_preds)

            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(cm, display_labels=["High Risk", "Low Risk"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.warning(f"Confusion matrix could not be drawn: {e}")