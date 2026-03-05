"""
explainer.py
Generates SHAP charts and returns base64 PNG.
"""

import shap
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64


# Load models
with open("models/xgb_risk.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("models/lgb_conv.pkl", "rb") as f:
    lgb_model = pickle.load(f)

with open("models/risk_features.pkl", "rb") as f:
    risk_features = pickle.load(f)

with open("models/conv_features.pkl", "rb") as f:
    conv_features = pickle.load(f)


risk_explainer = shap.TreeExplainer(xgb_model)
conv_explainer = shap.TreeExplainer(lgb_model)


# -------------------------
# RISK SHAP
# -------------------------

def get_risk_shap(X_row: np.ndarray, feature_names: list) -> dict:
    shap_vals = risk_explainer.shap_values(X_row)
    sv = np.array(shap_vals)

    if isinstance(shap_vals, list):
        sv = shap_vals[0][0]
    else:
        if sv.ndim == 3:
            sv = sv[0,:,0]
        elif sv.ndim == 2:
            sv = sv[0]

    result = {}
    for name, val in zip(feature_names, sv):
        result[name] = round(float(val),4)

    sorted_shap = dict(sorted(result.items(),
                              key=lambda x: abs(x[1]),
                              reverse=True))

    return dict(list(sorted_shap.items())[:5])


# -------------------------
# CONVERSION SHAP
# -------------------------

def get_conv_shap(X_row: np.ndarray, feature_names: list) -> dict:

    shap_vals = conv_explainer.shap_values(X_row)
    sv = np.array(shap_vals)

    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        if sv.ndim == 3:
            sv = sv[0,:,1]
        elif sv.ndim == 2:
            sv = sv[0]

    result = {}
    for name, val in zip(feature_names, sv):
        result[name] = round(float(val),4)

    sorted_shap = dict(sorted(result.items(),
                              key=lambda x: abs(x[1]),
                              reverse=True))

    return dict(list(sorted_shap.items())[:5])


# -------------------------
# SHAP PLOT
# -------------------------

def shap_to_base64(shap_dict: dict, title: str, color: str = "#00e5ff"):

    names = list(shap_dict.keys())
    vals  = list(shap_dict.values())

    colors = ["#10b981" if v > 0 else "#ef4444" for v in vals]

    fig, ax = plt.subplots(figsize=(6,3))

    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars = ax.barh(names, vals, color=colors)

    ax.axvline(0, color="#334155", lw=1)

    ax.set_title(title, color=color, fontsize=10)

    ax.tick_params(colors="#94a3b8")

    plt.tight_layout()

    buf = io.BytesIO()

    plt.savefig(buf,
                format="png",
                facecolor="#0d1117",
                bbox_inches="tight",
                dpi=120)

    plt.close()

    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")