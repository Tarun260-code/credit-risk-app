import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --------------------------------------------------
# 1. Create Synthetic Dataset (for Hackathon Demo)
# --------------------------------------------------

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    # Individual Features
    "age": np.random.randint(18, 60, n),
    "monthly_income": np.random.randint(10000, 100000, n),
    "employment_type": np.random.randint(0, 2, n),
    "years_job": np.random.randint(0, 15, n),
    "loan_amount": np.random.randint(50000, 500000, n),
    "loan_duration": np.random.randint(6, 60, n),
    "existing_emis": np.random.randint(0, 30000, n),
    "missed_payments": np.random.randint(0, 2, n),
    "savings_level": np.random.randint(0, 3, n),
    "city_tier": np.random.randint(0, 3, n),

    # MSME Features
    "business_age": np.random.randint(0, 20, n),
    "industry_type": np.random.randint(0, 4, n),
    "annual_turnover": np.random.randint(1000000, 20000000, n),
    "monthly_revenue": np.random.randint(100000, 2000000, n),
    "gst_registered": np.random.randint(0, 2, n),
    "existing_loans": np.random.randint(0, 1000000, n),
    "avg_balance": np.random.randint(10000, 500000, n),
    "cheque_bounce": np.random.randint(0, 2, n),
    "location_type": np.random.randint(0, 3, n)
})

# --------------------------------------------------
# 2. Create Risk Target (Smart Logic)
# --------------------------------------------------

risk_score = (
    data["loan_amount"] / (data["monthly_income"] + 1)
    + data["missed_payments"] * 2
    + data["cheque_bounce"] * 2
    + (data["existing_emis"] / 10000)
    - (data["savings_level"])
)

data["risk"] = (risk_score > np.median(risk_score)).astype(int)

# --------------------------------------------------
# 3. Train Model
# --------------------------------------------------

X = data.drop("risk", axis=1)
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------

y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print("AUC Score:", round(auc, 3))

# --------------------------------------------------
# 5. Save Model + Feature Names
# --------------------------------------------------

joblib.dump(model, "credit_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Model saved successfully.")
