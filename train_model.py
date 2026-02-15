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
df = pd.read_csv("german_credit.csv")
print("Shape:", df.shape)
print(df["Creditability"].value_counts())

# 2. Split
X = df.drop(columns=["Creditability"])
y = df["Creditability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE:", y_train.value_counts().to_dict())

# 4. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5. Train Models
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
rf.fit(X_train, y_train)

xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                    subsample=0.8, reg_lambda=2.0, random_state=42,
                    verbosity=0, eval_metric="logloss")
xgb.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_scaled, y_train)

print("All models trained")

# 6. Compare Models
models = {
    "Logistic Regression": (lr,  X_test_scaled),
    "Random Forest":       (rf,  X_test),
    "XGBoost":             (xgb, X_test),
    "KNN":                 (knn, X_test_scaled),
}

print("\n{:<25} {}".format("Model", "AUC"))
print("-" * 35)
for name, (model, X_eval) in models.items():
    auc = roc_auc_score(y_test, model.predict_proba(X_eval)[:, 1])
    print("{:<25} {:.3f}".format(name, auc))

# 7. ROC Curve
y_prob = xgb.predict_proba(X_test)[:, 1]
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

# 8. Confusion Matrix
y_pred = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print(classification_report(y_test, y_pred, target_names=["Bad Credit", "Good Credit"]))

# 9. Feature Importance
importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importance.plot(kind="barh", color="steelblue")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# 10. SHAP
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist(), show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

# 11. Save Model
joblib.dump(xgb, "model.pkl")
print("Model saved as model.pkl")

# Final AUC Summary
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