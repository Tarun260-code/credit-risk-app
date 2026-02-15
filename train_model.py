import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ------------------------------
# Generate Synthetic Indian Data
# ------------------------------

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "age": np.random.randint(21, 60, n),
    "monthly_income": np.random.randint(15000, 200000, n),
    "employment_years": np.random.randint(0, 30, n),
    "credit_score": np.random.randint(300, 900, n),
    "savings_balance": np.random.randint(0, 500000, n),
    "existing_loans": np.random.randint(0, 5, n),
    "loan_amount": np.random.randint(50000, 1000000, n),
    "loan_duration": np.random.randint(6, 60, n),
    "emi_ratio": np.random.uniform(0.1, 0.8, n),
    "digital_payment_frequency": np.random.randint(5, 200, n),
    "utility_bill_payment_score": np.random.randint(0, 100, n),
    "dependents": np.random.randint(0, 5, n),

    # MSME
    "business_age": np.random.randint(0, 20, n),
    "annual_turnover": np.random.randint(100000, 50000000, n),
    "gst_filing_score": np.random.randint(0, 100, n),
    "average_monthly_cashflow": np.random.randint(20000, 1000000, n),
    "existing_business_loans": np.random.randint(0, 5, n),
    "sector_risk_score": np.random.uniform(0.1, 1.0, n),
    "supplier_payment_score": np.random.randint(0, 100, n),
    "upi_transaction_volume": np.random.randint(10, 1000, n),
    "inventory_turnover_ratio": np.random.uniform(0.5, 10.0, n),
    "profit_margin": np.random.uniform(0.01, 0.4, n),
    "credit_history_score": np.random.randint(0, 100, n)
})

# Target variable
data["default"] = (
    (data["credit_score"] < 500) |
    (data["emi_ratio"] > 0.6) |
    (data["gst_filing_score"] < 40)
).astype(int)

X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

joblib.dump(model, "credit_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
