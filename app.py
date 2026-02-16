import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import datetime
import re

matplotlib.use("Agg")

st.set_page_config(page_title="AI Credit Risk System", layout="centered")
st.title("🏦 AI Credit Risk Evaluation")
st.write("Select applicant type to begin.")

# ─────────────────────────────────────────
# Load / Train Model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    import os, joblib
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE

    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl"), True

    if not os.path.exists("cs-training.csv"):
        return None, False

    df = pd.read_csv("cs-training.csv", index_col=0)
    df.dropna(subset=["MonthlyIncome", "NumberOfDependents"], inplace=True)
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))
    df = df[df["age"] >= 18]

    df["TotalLate"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )
    df["DelinquencyScore"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1 +
        df["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
        df["NumberOfTimes90DaysLate"]              * 3
    )
    df["DebtToIncome"]       = df["DebtRatio"] / (df["MonthlyIncome"] + 1)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=2.0, random_state=42, verbosity=0, eval_metric="logloss"
    )
    model.fit(X_res.values, y_res)
    joblib.dump(model, "model.pkl")
    return model, True

model, model_loaded = load_model()

# ─────────────────────────────────────────
# Applicant Type
# ─────────────────────────────────────────
applicant_type = st.radio("Applicant Type", ["Individual", "MSME"], horizontal=True)
st.divider()

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def validate_pan(pan):
    return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", pan.strip().upper()))

def compute_credit_score(prob):
    return max(300, min(850, int(850 - prob * 550)))

def score_band(score):
    if score >= 750: return "Excellent", "🟢"
    if score >= 700: return "Good",      "🟡"
    if score >= 650: return "Fair",      "🟠"
    if score >= 600: return "Poor",      "🔴"
    return "Very Poor", "🔴"

# ─────────────────────────────────────────
# Risk Logic — Individual
# ─────────────────────────────────────────
def individual_risk(age, monthly_income, required_loan, missed_payment,
                    existing_loan, years_employment, has_insurance):
    prob = 0.15

    if age < 25:            prob += 0.10
    elif age > 55:          prob += 0.05

    lti = required_loan / (monthly_income + 1)
    if lti > 10:            prob += 0.20
    elif lti > 5:           prob += 0.10
    elif lti < 2:           prob -= 0.05

    if missed_payment == "Yes, multiple times":  prob += 0.25
    elif missed_payment == "Yes, once or twice": prob += 0.12

    if existing_loan == "Yes, heavy burden":     prob += 0.15
    elif existing_loan == "Yes, manageable":     prob += 0.05

    if years_employment < 1:    prob += 0.15
    elif years_employment < 3:  prob += 0.05
    elif years_employment >= 5: prob -= 0.05

    if has_insurance:           prob -= 0.05

    return max(0.02, min(0.97, prob))

# ─────────────────────────────────────────
# Risk Logic — MSME
# ─────────────────────────────────────────
def msme_risk(owner_age, monthly_income, business_age, gst_registered,
              monthly_revenue, profit_margin_pct, required_loan,
              missed_payment, existing_loan, years_employment, has_insurance):
    prob = 0.20

    if business_age < 1:       prob += 0.20
    elif business_age < 3:     prob += 0.10
    elif business_age >= 5:    prob -= 0.08

    if not gst_registered:     prob += 0.12

    ltr = required_loan / (monthly_revenue + 1)
    if ltr > 12:   prob += 0.20
    elif ltr > 6:  prob += 0.10
    elif ltr < 2:  prob -= 0.05

    if profit_margin_pct < 5:     prob += 0.15
    elif profit_margin_pct < 15:  prob += 0.05
    elif profit_margin_pct >= 25: prob -= 0.08

    if missed_payment == "Yes, multiple times":  prob += 0.20
    elif missed_payment == "Yes, once or twice": prob += 0.10

    if existing_loan == "Yes, heavy burden":     prob += 0.12
    elif existing_loan == "Yes, manageable":     prob += 0.04

    if years_employment < 1:    prob += 0.08
    elif years_employment >= 5: prob -= 0.04

    if has_insurance:           prob -= 0.05

    return max(0.02, min(0.97, prob))

# ─────────────────────────────────────────
# Gauge Chart
# ─────────────────────────────────────────
def draw_gauge(credit_score):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    bands = [
        (300, 500, "#d32f2f", "Very Poor"),
        (500, 580, "#e57373", "Poor"),
        (580, 650, "#ff9800", "Fair"),
        (650, 700, "#fff176", "Good"),
        (700, 750, "#66bb6a", "Very Good"),
        (750, 850, "#2e7d32", "Excellent"),
    ]
    for lo, hi, col, lbl in bands:
        ax.barh(0, hi - lo, left=lo, color=col, height=0.4)
        ax.text((lo + hi) / 2, -0.32, lbl, ha="center", fontsize=7)
    ax.axvline(credit_score, color="black", linewidth=2.5, linestyle="--")
    ax.text(credit_score, 0.28, str(credit_score),
            ha="center", fontweight="bold", fontsize=12)
    ax.set_xlim(300, 850)
    ax.set_yticks([])
    ax.set_xlabel("Credit Score")
    ax.set_title("Credit Score Range")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# ML Charts — ROC, Confusion Matrix, SHAP
# ─────────────────────────────────────────
def show_ml_charts(input_df):
    if not model_loaded or model is None:
        st.info("ℹ️ Model not available — ML charts unavailable.")
        return

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (roc_curve, roc_auc_score,
                                     confusion_matrix, ConfusionMatrixDisplay)
        import shap

        # Load & prep training data
        df = pd.read_csv("cs-training.csv", index_col=0)
        df.dropna(subset=["MonthlyIncome", "NumberOfDependents"], inplace=True)
        for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
            df[col] = df[col].clip(upper=df[col].quantile(0.99))
        df = df[df["age"] >= 18]

        df["TotalLate"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"] +
            df["NumberOfTime60-89DaysPastDueNotWorse"] +
            df["NumberOfTimes90DaysLate"]
        )
        df["DelinquencyScore"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"] * 1 +
            df["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
            df["NumberOfTimes90DaysLate"]              * 3
        )
        df["DebtToIncome"]       = df["DebtRatio"] / (df["MonthlyIncome"] + 1)
        df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

        X = df.drop(columns=["SeriousDlqin2yrs"])
        y = df["SeriousDlqin2yrs"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        probs = model.predict_proba(X_test.values)[:, 1]
        preds = model.predict(X_test.values)

        # ── ROC + Confusion Matrix ──────────────
        st.subheader("📈 Model Performance")
        col_roc, col_cm = st.columns(2)

        with col_roc:
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc_val      = roc_auc_score(y_test, probs)
            fig_r, ax_r  = plt.subplots(figsize=(5, 4))
            ax_r.plot(fpr, tpr, color="steelblue", label=f"AUC = {auc_val:.3f}")
            ax_r.plot([0, 1], [0, 1], "--", color="gray")
            ax_r.set_xlabel("False Positive Rate")
            ax_r.set_ylabel("True Positive Rate")
            ax_r.set_title("ROC Curve")
            ax_r.legend()
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)

        with col_cm:
            cm_val = confusion_matrix(y_test, preds)
            fig_c, ax_c = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(
                cm_val, display_labels=["No Default", "Default"]
            ).plot(ax=ax_c, colorbar=False, cmap="Blues")
            ax_c.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

        # ── SHAP Waterfall ──────────────────────
        st.subheader("🔍 Why this prediction? (SHAP)")
        try:
            input_aligned = input_df.reindex(columns=X.columns, fill_value=0)

            # Force clean numpy float64 — fixes '[5E-1]' string bug
            input_array = input_aligned.values.astype(np.float64)

            explainer = shap.TreeExplainer(model)
            shap_vals = np.array(
                explainer.shap_values(input_array), dtype=np.float64
            )

            # expected_value can return as string '[5E-1]' — handle all cases
            raw_base = explainer.expected_value
            if isinstance(raw_base, (list, np.ndarray)):
                base_val = float(np.array(raw_base).flat[0])
            else:
                base_val = float(str(raw_base).strip("[]"))

            explanation = shap.Explanation(
                values        = shap_vals[0],
                base_values   = base_val,
                data          = input_array[0],
                feature_names = X.columns.tolist(),
            )
            fig_s, _ = plt.subplots()
            shap.waterfall_plot(explanation, show=False)
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

        except Exception as e:
            st.warning(f"SHAP could not run: {e}")

    except FileNotFoundError:
        st.info("ℹ️ cs-training.csv not found — ML charts unavailable.")
    except Exception as e:
        st.warning(f"ML charts error: {e}")

# ─────────────────────────────────────────
# PDF Generator
# ─────────────────────────────────────────
def generate_pdf(data: dict, prob, prediction, credit_score, band, app_type):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle("t", parent=styles["Title"],
                              fontSize=18, spaceAfter=4, alignment=TA_CENTER)
    sub_s   = ParagraphStyle("s", parent=styles["Normal"],
                              fontSize=10, textColor=colors.grey, alignment=TA_CENTER)
    head_s  = ParagraphStyle("h", parent=styles["Heading2"],
                              fontSize=12, spaceBefore=12, spaceAfter=4)
    disc_s  = ParagraphStyle("d", parent=styles["Normal"],
                              fontSize=8, textColor=colors.grey, alignment=TA_CENTER)

    now    = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    result = "LOW RISK ✓" if prediction == "Low Risk" else "HIGH RISK ✗"
    rcol   = colors.green if prediction == "Low Risk" else colors.red

    elems = []
    elems.append(Paragraph("AI Credit Risk Evaluation Report", title_s))
    elems.append(Paragraph(f"Type: {app_type}  |  Generated on {now}", sub_s))
    elems.append(Spacer(1, 0.3*cm))
    elems.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elems.append(Spacer(1, 0.3*cm))

    elems.append(Paragraph("Credit Summary", head_s))
    s_tbl = Table(
        [["Credit Score", "Band", "Default Probability", "Decision"],
         [str(credit_score), band, f"{round(prob*100,1)} %", result]],
        colWidths=[4*cm, 4*cm, 5*cm, 5*cm]
    )
    s_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",   (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",   (0,0),(-1,-1), 10),
        ("ALIGN",      (0,0),(-1,-1),"CENTER"),
        ("GRID",       (0,0),(-1,-1), 0.5, colors.lightgrey),
        ("PADDING",    (0,0),(-1,-1), 8),
        ("TEXTCOLOR",  (3,1),(3,1),   rcol),
        ("FONTNAME",   (3,1),(3,1),   "Helvetica-Bold"),
    ]))
    elems.append(s_tbl)
    elems.append(Spacer(1, 0.3*cm))

    elems.append(Paragraph("Applicant Details", head_s))
    d_tbl = Table(
        [["Field", "Value"]] + [[k, str(v)] for k, v in data.items()],
        colWidths=[9*cm, 9*cm]
    )
    d_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#EEF2FF")),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F8F9FF")]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.lightgrey),
        ("PADDING",       (0,0),(-1,-1), 6),
    ]))
    elems.append(d_tbl)
    elems.append(Spacer(1, 0.3*cm))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elems.append(Spacer(1, 0.2*cm))
    elems.append(Paragraph(
        "Disclaimer: This report is AI-generated for informational purposes only "
        "and does not constitute a formal credit decision.", disc_s))

    doc.build(elems)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────
# Shared Result Display
# ─────────────────────────────────────────
def show_results(prob, report_data, pan, app_type, input_df):
    credit_score = compute_credit_score(prob)
    band, emoji  = score_band(credit_score)
    prediction   = "Low Risk" if prob < 0.5 else "High Risk"

    st.divider()
    st.subheader("📊 Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Credit Score", credit_score)
    c2.metric("Default Probability", f"{round(prob*100,1)} %")
    c3.metric("Risk Band", f"{emoji} {band}")

    if prediction == "Low Risk":
        st.success("✅ LOW RISK — Likely to repay.")
    else:
        st.error("⚠️ HIGH RISK — Risk of default detected.")

    st.pyplot(draw_gauge(credit_score))
    plt.close()

    st.divider()
    show_ml_charts(input_df)
    st.divider()

    st.subheader("📄 Download Report")
    pdf = generate_pdf(report_data, prob, prediction, credit_score, band, app_type)
    if pdf:
        st.download_button(
            "⬇️ Download Credit Report (PDF)", pdf,
            file_name=f"credit_report_{pan.upper()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.info("💡 Run: pip install reportlab  to enable PDF download")

# ─────────────────────────────────────────
# Build Input DataFrame for SHAP
# ─────────────────────────────────────────
def build_input_df(age, monthly_income, required_loan,
                   missed_payment, existing_loan):
    late_flag = 1 if missed_payment != "No" else 0
    severe    = 1 if missed_payment == "Yes, multiple times" else 0
    return pd.DataFrame([[
        min(monthly_income / (monthly_income + 1) * 0.3, 1.0),  # RevolvingUtilization
        age,                                                       # age
        required_loan / (monthly_income * 12 + 1),                # DebtRatio
        monthly_income,                                            # MonthlyIncome
        1 if existing_loan != "No" else 0,                        # OpenCreditLines
        severe,                                                    # 90DaysLate
        0,                                                         # RealEstateLoans
        late_flag,                                                 # 30-59DaysLate
        0,                                                         # Dependents
        0,                                                         # 60-89DaysLate
        late_flag,                                                 # TotalLate
        late_flag + severe,                                        # DelinquencyScore
        required_loan / (monthly_income * 12 + 1) / (monthly_income + 1),  # DebtToIncome
        monthly_income,                                            # IncomePerDependent
    ]], columns=[
        "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
        "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfDependents",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "TotalLate", "DelinquencyScore", "DebtToIncome", "IncomePerDependent",
    ])

# ─────────────────────────────────────────
# INDIVIDUAL FORM
# ─────────────────────────────────────────
if applicant_type == "Individual":
    st.subheader("👤 Individual Details")

    col1, col2 = st.columns(2)
    with col1:
        name           = st.text_input("Full Name")
        pan            = st.text_input("PAN Number", max_chars=10,
                                       help="Format: ABCDE1234F")
        age            = st.number_input("Age (years)", 18, 100, 30)
        monthly_income = st.number_input("Monthly Income (₹)", 0, 10000000,
                                         40000, step=1000)
    with col2:
        required_loan    = st.number_input("Required Loan Amount (₹)", 0,
                                           100000000, 200000, step=5000)
        missed_payment   = st.selectbox("Have you missed any payment?",
                                        ["No", "Yes, once or twice",
                                         "Yes, multiple times"])
        existing_loan    = st.selectbox("Existing Loan?",
                                        ["No", "Yes, manageable",
                                         "Yes, heavy burden"])
        years_employment = st.number_input("Years of Employment", 0, 50, 3)
        has_insurance    = st.checkbox("Have Insurance?")

    if st.button("🔍 Evaluate", use_container_width=True):
        if not validate_pan(pan):
            st.error("❌ Invalid PAN. Format: ABCDE1234F")
            st.stop()
        if not name.strip():
            st.error("❌ Please enter applicant name.")
            st.stop()

        prob     = individual_risk(age, monthly_income, required_loan,
                                   missed_payment, existing_loan,
                                   years_employment, has_insurance)
        input_df = build_input_df(age, monthly_income, required_loan,
                                  missed_payment, existing_loan)

        report_data = {
            "Name": name, "PAN": pan.upper(), "Age": age,
            "Monthly Income":    f"₹ {monthly_income:,}",
            "Required Loan":     f"₹ {required_loan:,}",
            "Missed Payment":    missed_payment,
            "Existing Loan":     existing_loan,
            "Years of Employment": years_employment,
            "Insurance":         "Yes" if has_insurance else "No",
        }
        show_results(prob, report_data, pan, "Individual", input_df)

# ─────────────────────────────────────────
# MSME FORM
# ─────────────────────────────────────────
else:
    st.subheader("🏢 MSME Details")

    st.markdown("**Owner / Proprietor Info**")
    col1, col2 = st.columns(2)
    with col1:
        name           = st.text_input("Owner Name")
        pan            = st.text_input("PAN Number", max_chars=10,
                                       help="Format: ABCDE1234F")
        owner_age      = st.number_input("Owner Age (years)", 18, 100, 35)
        monthly_income = st.number_input("Owner Monthly Income (₹)", 0,
                                         10000000, 50000, step=1000)
    with col2:
        missed_payment   = st.selectbox("Missed any payment?",
                                        ["No", "Yes, once or twice",
                                         "Yes, multiple times"])
        existing_loan    = st.selectbox("Existing Loan?",
                                        ["No", "Yes, manageable",
                                         "Yes, heavy burden"])
        years_employment = st.number_input("Years Running Business", 0, 50, 3)
        has_insurance    = st.checkbox("Business Insurance?")

    st.markdown("**Business Info**")
    col3, col4 = st.columns(2)
    with col3:
        business_age   = st.number_input("Business Age (years)", 0, 100, 3)
        gst_registered = st.radio("GST Registered?", ["Yes", "No"],
                                   horizontal=True) == "Yes"
        required_loan  = st.number_input("Required Loan Amount (₹)", 0,
                                         100000000, 500000, step=10000)
    with col4:
        monthly_revenue = st.number_input("Monthly Revenue (₹)", 0,
                                          100000000, 200000, step=5000)
        profit_margin   = st.slider("Profit Margin (%)", 0, 100, 15)

    if st.button("🔍 Evaluate", use_container_width=True):
        if not validate_pan(pan):
            st.error("❌ Invalid PAN. Format: ABCDE1234F")
            st.stop()
        if not name.strip():
            st.error("❌ Please enter owner name.")
            st.stop()

        prob     = msme_risk(owner_age, monthly_income, business_age,
                             gst_registered, monthly_revenue, profit_margin,
                             required_loan, missed_payment, existing_loan,
                             years_employment, has_insurance)
        input_df = build_input_df(owner_age, monthly_income, required_loan,
                                  missed_payment, existing_loan)

        report_data = {
            "Owner Name":          name,
            "PAN":                 pan.upper(),
            "Owner Age":           owner_age,
            "Owner Monthly Income":f"₹ {monthly_income:,}",
            "Business Age":        f"{business_age} years",
            "GST Registered":      "Yes" if gst_registered else "No",
            "Monthly Revenue":     f"₹ {monthly_revenue:,}",
            "Profit Margin":       f"{profit_margin} %",
            "Required Loan":       f"₹ {required_loan:,}",
            "Missed Payment":      missed_payment,
            "Existing Loan":       existing_loan,
            "Years Running Business": years_employment,
            "Insurance":           "Yes" if has_insurance else "No",
        }
        show_results(prob, report_data, pan, "MSME", input_df)


