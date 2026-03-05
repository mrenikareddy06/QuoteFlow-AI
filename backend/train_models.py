"""
train_models.py
Run this ONCE before starting the server:
    python train_models.py
It reads data/quotes.csv, trains both models, saves them to models/
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/quotes.csv")
print(f"Loaded {len(df)} rows")

# ── Clean ────────────────────────────────────────────────────
df = df.dropna(subset=["Policy_Bind"])
df["Policy_Bind_Binary"] = (df["Policy_Bind"] == "Yes").astype(int)

# ── Encode categorical columns ───────────────────────────────
cat_cols = [
    "Agent_Type", "Region", "Policy_Type", "Gender",
    "Marital_Status", "Education", "Sal_Range",
    "Coverage", "Veh_Usage", "Annual_Miles_Range",
    "Vehicl_Cost_Range", "Re_Quote"
]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
os.makedirs("models", exist_ok=True)
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("Saved encoders")

# ════════════════════════════════════════════════════════════
# MODEL 1 — XGBoost Risk Profiler
# Features: accident/citation/driving history
# ════════════════════════════════════════════════════════════
print("\nTraining Agent 1 — XGBoost Risk Profiler...")

risk_features = [
    "Prev_Accidents", "Prev_Citations", "Driving_Exp",
    "Driver_Age", "Veh_Usage_enc", "Annual_Miles_Range_enc",
    "HH_Vehicles", "HH_Drivers"
]

# Create risk label (HIGH if accidents OR citations)
df["risk_label"] = 0  # LOW
df.loc[(df["Prev_Accidents"] == 1) | (df["Prev_Citations"] == 1), "risk_label"] = 1  # MED
df.loc[(df["Prev_Accidents"] == 1) & (df["Prev_Citations"] == 1), "risk_label"] = 2  # HIGH

X_risk = df[risk_features].fillna(0)
y_risk = df["risk_label"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_risk, y_risk, test_size=0.2, random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train_r, y_train_r)

print("XGBoost Risk Report:")
print(classification_report(y_test_r, xgb_model.predict(X_test_r),
      target_names=["LOW", "MED", "HIGH"]))

with open("models/xgb_risk.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open("models/risk_features.pkl", "wb") as f:
    pickle.dump(risk_features, f)
print("Saved XGBoost model")

# ════════════════════════════════════════════════════════════
# MODEL 2 — LightGBM Conversion Predictor
# Features: quote-level and behavioral signals
# ════════════════════════════════════════════════════════════
print("\nTraining Agent 2 — LightGBM Conversion Predictor...")

conv_features = [
    "Re_Quote_enc", "Coverage_enc", "Agent_Type_enc",
    "Region_enc", "Sal_Range_enc", "HH_Drivers",
    "Vehicl_Cost_Range_enc", "Policy_Type_enc",
    "Prev_Accidents", "Prev_Citations", "Driver_Age",
    "Quoted_Premium", "HH_Vehicles"
]

X_conv = df[conv_features].fillna(0)
y_conv = df["Policy_Bind_Binary"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_conv, y_conv, test_size=0.2, random_state=42
)

# SMOTE — Layer 1 of 3-layer imbalance fix
print(f"Before SMOTE — 0: {sum(y_train_c==0)}, 1: {sum(y_train_c==1)}")
smote = SMOTE(random_state=42)
X_train_c_bal, y_train_c_bal = smote.fit_resample(X_train_c, y_train_c)
print(f"After SMOTE  — 0: {sum(y_train_c_bal==0)}, 1: {sum(y_train_c_bal==1)}")

# Layer 2: class_weight='balanced'
lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    class_weight="balanced",   # Layer 2
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train_c_bal, y_train_c_bal)

# Layer 3: threshold tuned to 0.35 (handled at inference time)
print("LightGBM Conversion Report:")
from sklearn.metrics import classification_report as cr
proba = lgb_model.predict_proba(X_test_c)[:, 1]
preds = (proba >= 0.35).astype(int)  # Layer 3 threshold
print(cr(y_test_c, preds, target_names=["NO_BIND", "BIND"]))

with open("models/lgb_conv.pkl", "wb") as f:
    pickle.dump(lgb_model, f)
with open("models/conv_features.pkl", "wb") as f:
    pickle.dump(conv_features, f)

print("\nAll models trained and saved to models/")
print("Now run: uvicorn main:app --reload")