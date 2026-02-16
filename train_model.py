import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import joblib

# 1. Load Data
df = pd.read_csv("cs-training.csv", index_col=0)
print("Shape:", df.shape)
print(df["SeriousDlqin2yrs"].value_counts())

# 2. Clean
df.dropna(subset=["MonthlyIncome", "NumberOfDependents"], inplace=True)

for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

df = df[df["age"] >= 18]
print("Shape after cleaning:", df.shape)

# 3. Feature Engineering
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

# 4. Split
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", y_train_res.value_counts().to_dict())

# 6. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

# 7. Train Models
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_res)

rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)
rf.fit(X_train_res, y_train_res)

xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=2.0, random_state=42, verbosity=0, eval_metric="logloss"
)
xgb.fit(X_train_res.values, y_train_res)

knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
knn.fit(X_train_scaled, y_train_res)

print("All models trained")

# 8. Compare Models
models = {
    "Logistic Regression": (lr,  X_test_scaled),
    "Random Forest":       (rf,  X_test),
    "XGBoost":             (xgb, X_test.values),
    "KNN":                 (knn, X_test_scaled),
}

print("\n{:<25} {}".format("Model", "AUC"))
print("-" * 35)
for name, (model, X_eval) in models.items():
    auc = roc_auc_score(y_test, model.predict_proba(X_eval)[:, 1])
    print("{:<25} {:.3f}".format(name, auc))

# 9. ROC Curve
y_prob = xgb.predict_proba(X_test.values)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"XGBoost AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# 10. Confusion Matrix
y_pred = xgb.predict(X_test.values)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Delinquency", "Delinquency"],
            yticklabels=["No Delinquency", "Delinquency"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print(classification_report(y_test, y_pred,
                             target_names=["No Delinquency", "Delinquency"]))

# 11. Feature Importance
importance = pd.Series(
    xgb.feature_importances_, index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importance.plot(kind="barh", color="steelblue")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# 12. SHAP
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test.values)

shap.summary_plot(shap_values, X_test.values,
                  feature_names=X.columns.tolist(), show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

# 13. Save Model
joblib.dump(xgb, "model.pkl")
print("Model saved as model.pkl")

# 14. Score cs-test.csv → submission
df_test = pd.read_csv("cs-test.csv", index_col=0)

# Same feature engineering
df_test["TotalLate"] = (
    df_test["NumberOfTime30-59DaysPastDueNotWorse"] +
    df_test["NumberOfTime60-89DaysPastDueNotWorse"] +
    df_test["NumberOfTimes90DaysLate"]
)
df_test["DelinquencyScore"] = (
    df_test["NumberOfTime30-59DaysPastDueNotWorse"] * 1 +
    df_test["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
    df_test["NumberOfTimes90DaysLate"]              * 3
)
df_test["DebtToIncome"]       = df_test["DebtRatio"] / (df_test["MonthlyIncome"] + 1)
df_test["IncomePerDependent"] = df_test["MonthlyIncome"] / (df_test["NumberOfDependents"] + 1)

# Same caps from training
for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
    if col in df_test.columns:
        df_test[col] = df_test[col].clip(upper=df[col].quantile(0.99))

df_test["MonthlyIncome"]      = df_test["MonthlyIncome"].fillna(df_test["MonthlyIncome"].median())
df_test["NumberOfDependents"] = df_test["NumberOfDependents"].fillna(0)

# Drop placeholder column, align to training features
df_test = df_test.drop(columns=["SeriousDlqin2yrs"], errors="ignore")
df_test = df_test[X.columns]

submission = pd.DataFrame({
    "Id":          df_test.index,
    "Probability": xgb.predict_proba(df_test.values)[:, 1]
})
submission.to_csv("sampleEntry.csv", index=False)
print("Submission saved as sampleEntry.csv")

# Final Summary
print("\n" + "=" * 40)
print("Final Model Performance Summary")
print("=" * 40)
print(f"Model     : XGBoost")
print(f"Test AUC  : {auc:.3f}")
if auc >= 0.90:
    print("Result    : Excellent — but check for overfitting")
elif auc >= 0.80:
    print("Result    : Good — model generalises well")
elif auc >= 0.70:
    print("Result    : Fair — consider tuning or more data")
else:
    print("Result    : Poor — model needs improvement")
print("=" * 40)