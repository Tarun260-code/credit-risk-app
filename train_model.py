import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("german_credit.csv")

print("Columns:", df.columns)

# ==============================
# 2. Define Target
# ==============================
TARGET = "Creditability"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==============================
# 3. Encode Categorical Columns
# ==============================
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# ==============================
# 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5. Define XGBoost Model
# ==============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

# ==============================
# 6. Train Model
# ==============================
model.fit(X_train, y_train)

print("Model training completed.")

# ==============================
# 7. Evaluate Model
# ==============================
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print("AUC Score:", round(auc, 4))

# ==============================
# 8. Save Model
# ==============================
joblib.dump(model, "credit_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Model and feature names saved.")
