"""
pipeline.py
LangGraph 4-agent pipeline.
Each agent reads the shared state, adds its output, passes forward.
"""

import pickle
import numpy as np
import requests
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from explainer import (
    get_risk_shap, get_conv_shap,
    shap_to_base64, risk_features, conv_features,
    xgb_model, lgb_model
)

# ── Load encoders ─────────────────────────────────────────────
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

OLLAMA_URL = "http://localhost:11434/api/generate"

# ══════════════════════════════════════════════════════════════
# STATE SCHEMA — typed dict, each agent adds its own fields
# ══════════════════════════════════════════════════════════════
class QuoteState(TypedDict):
    # ── INPUT ──────────────────────────────────────────
    quote_id:            str
    agent_type:          str
    region:              str
    policy_type:         str
    hh_vehicles:         int
    hh_drivers:          int
    driver_age:          int
    driving_exp:         int
    prev_accidents:      int
    prev_citations:      int
    gender:              str
    marital_status:      str
    education:           str
    sal_range:           str
    coverage:            str
    veh_usage:           str
    annual_miles_range:  str
    vehicl_cost_range:   str
    re_quote:            str
    quoted_premium:      float
    q_valid_dt:          str

    # ── AGENT 1 OUTPUT ─────────────────────────────────
    risk_tier:           Optional[str]      # LOW / MEDIUM / HIGH
    risk_score:          Optional[float]
    risk_shap:           Optional[dict]
    risk_shap_chart:     Optional[str]      # base64 PNG

    # ── AGENT 2 OUTPUT ─────────────────────────────────
    bind_score:          Optional[float]    # 0.0 – 1.0
    bind_pct:            Optional[int]      # 0 – 100
    conv_shap:           Optional[dict]
    conv_shap_chart:     Optional[str]

    # ── AGENT 3 OUTPUT ─────────────────────────────────
    premium_flag:        Optional[str]      # BLOCKER / OK
    premium_ratio:       Optional[float]    # premium as % of salary
    llm_reasoning:       Optional[str]

    # ── AGENT 4 OUTPUT ─────────────────────────────────
    final_decision:      Optional[str]      # AUTO_APPROVE / FOLLOW_UP / ESCALATE
    escalation_reason:   Optional[str]
    llm_summary:         Optional[str]
    audit_trail:         Optional[list]


# ── Helper: encode a categorical value ───────────────────────
def enc(col: str, val: str) -> int:
    try:
        return int(encoders[col].transform([str(val)])[0])
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════
# AGENT 1 — RISK PROFILER (XGBoost)
# ══════════════════════════════════════════════════════════════
def agent1_risk_profiler(state: QuoteState) -> QuoteState:
    X = np.array([[
        state["prev_accidents"],
        state["prev_citations"],
        state["driving_exp"],
        state["driver_age"],
        enc("Veh_Usage",          state["veh_usage"]),
        enc("Annual_Miles_Range", state["annual_miles_range"]),
        state["hh_vehicles"],
        state["hh_drivers"],
    ]])

    pred  = int(xgb_model.predict(X)[0])
    proba = xgb_model.predict_proba(X)[0]

    tier_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    tier = tier_map[pred]

    shap_dict  = get_risk_shap(X, risk_features)
    shap_chart = shap_to_base64(shap_dict, "Risk Profiler — SHAP", "#00e5ff")

    return {
        **state,
        "risk_tier":       tier,
        "risk_score":      round(float(max(proba)), 3),
        "risk_shap":       shap_dict,
        "risk_shap_chart": shap_chart,
    }


# ══════════════════════════════════════════════════════════════
# AGENT 2 — CONVERSION PREDICTOR (LightGBM + SMOTE)
# ══════════════════════════════════════════════════════════════
def agent2_conversion_predictor(state: QuoteState) -> QuoteState:
    X = np.array([[
        enc("Re_Quote",           state["re_quote"]),
        enc("Coverage",           state["coverage"]),
        enc("Agent_Type",         state["agent_type"]),
        enc("Region",             state["region"]),
        enc("Sal_Range",          state["sal_range"]),
        state["hh_drivers"],
        enc("Vehicl_Cost_Range",  state["vehicl_cost_range"]),
        enc("Policy_Type",        state["policy_type"]),
        state["prev_accidents"],
        state["prev_citations"],
        state["driver_age"],
        state["quoted_premium"],
        state["hh_vehicles"],
    ]])

    proba      = lgb_model.predict_proba(X)[0][1]
    bind_score = round(float(proba), 3)
    # Layer 3: threshold 0.35 instead of default 0.5
    bind_pct   = int(bind_score * 100)

    shap_dict  = get_conv_shap(X, conv_features)
    shap_chart = shap_to_base64(shap_dict, "Conversion Predictor — SHAP", "#f59e0b")

    return {
        **state,
        "bind_score":      bind_score,
        "bind_pct":        bind_pct,
        "conv_shap":       shap_dict,
        "conv_shap_chart": shap_chart,
    }


# ══════════════════════════════════════════════════════════════
# AGENT 3 — PREMIUM ADVISOR (Llama 3.1 via Ollama)
# ══════════════════════════════════════════════════════════════

# Salary bracket midpoints (for ratio calculation)
SAL_MIDPOINTS = {
    "<= $ 25 K":            25000,
    "> $ 25 K <= $ 40 K":   32500,
    "> $ 40 K <= $ 60 K":   50000,
    "> $ 60 K <= $ 90 K":   75000,
    "> $ 90 K":             100000,
}

def agent3_premium_advisor(state: QuoteState) -> QuoteState:
    premium  = state["quoted_premium"]
    sal_key  = state["sal_range"]
    sal_mid  = SAL_MIDPOINTS.get(sal_key, 40000)
    ratio    = round((premium / sal_mid) * 100, 2)

    # Rule: ratio > 2.5% = potential blocker
    likely_blocker = ratio > 2.5

    prompt = f"""You are an insurance analyst. Analyze this quote briefly.

Customer salary range: {sal_key} (midpoint ~${sal_mid:,}/year)
Quoted premium: ${premium:.2f}/year  
Premium as % of salary: {ratio}%
Coverage type: {state['coverage']}
Re-quote (returned customer): {state['re_quote']}
Bind probability score: {state['bind_pct']}%

Industry threshold: premium should be under 2.5% of annual salary.

Answer in exactly 2 sentences:
1. Is the premium a conversion BLOCKER? Why?
2. What do you recommend?"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": "llama3.1", "prompt": prompt, "stream": False},
            timeout=30
        )
        reasoning = resp.json().get("response", "LLM unavailable").strip()
    except Exception as e:
        reasoning = f"Offline mode: Premium ratio {ratio}% {'exceeds' if likely_blocker else 'within'} 2.5% threshold."

    flag = "BLOCKER" if likely_blocker else "OK"

    return {
        **state,
        "premium_flag":   flag,
        "premium_ratio":  ratio,
        "llm_reasoning":  reasoning,
    }


# ══════════════════════════════════════════════════════════════
# AGENT 4 — DECISION ROUTER
# ══════════════════════════════════════════════════════════════
def agent4_decision_router(state: QuoteState) -> QuoteState:
    risk       = state["risk_tier"]
    bind       = state["bind_score"]
    flag       = state["premium_flag"]
    re_quote   = state["re_quote"]
    bind_score = state.get("bind_score", 0)

    # ── Escalation triggers ──────────────────────────
    escalation_reasons = []

    if risk == "HIGH" and bind < 0.50:
        escalation_reasons.append("HIGH risk + bind score below 50%")
    if bind < 0.35:
        escalation_reasons.append("Bind score critically low (<35%)")
    if flag == "BLOCKER" and re_quote == "Yes" and risk == "HIGH":
        escalation_reasons.append("Triple signal: premium blocker + re-quote + HIGH risk")

    # ── Routing logic ────────────────────────────────
    if escalation_reasons:
        decision = "ESCALATE"
    elif risk == "LOW" and bind >= 0.65 and flag == "OK":
        decision = "AUTO_APPROVE"
    else:
        decision = "FOLLOW_UP"

    # ── LLM summary for escalated cases ─────────────
    llm_summary = ""
    if decision == "ESCALATE":
        prompt = f"""Write a 2-sentence escalation summary for an underwriter.

Quote ID: {state['quote_id']}
Risk Tier: {risk}
Bind Score: {state['bind_pct']}%
Premium Flag: {flag} (ratio: {state.get('premium_ratio', 'N/A')}%)
Escalation Reason: {', '.join(escalation_reasons)}
Region: {state['region']}, Agent Type: {state['agent_type']}

Be concise and professional."""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": "llama3.1", "prompt": prompt, "stream": False},
                timeout=30
            )
            llm_summary = resp.json().get("response", "").strip()
        except Exception:
            llm_summary = f"Quote {state['quote_id']} escalated: {', '.join(escalation_reasons)}."

    audit = [
        {"agent": "Risk Profiler",       "output": risk,                    "model": "XGBoost"},
        {"agent": "Conversion Predictor","output": f"{state['bind_pct']}%", "model": "LightGBM"},
        {"agent": "Premium Advisor",     "output": flag,                    "model": "Llama 3.1"},
        {"agent": "Decision Router",     "output": decision,                "model": "Rules+LLM"},
    ]

    return {
        **state,
        "final_decision":    decision,
        "escalation_reason": "; ".join(escalation_reasons) if escalation_reasons else "None",
        "llm_summary":       llm_summary,
        "audit_trail":       audit,
    }


# ══════════════════════════════════════════════════════════════
# BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════
def build_pipeline():
    graph = StateGraph(QuoteState)

    graph.add_node("agent1", agent1_risk_profiler)
    graph.add_node("agent2", agent2_conversion_predictor)
    graph.add_node("agent3", agent3_premium_advisor)
    graph.add_node("agent4", agent4_decision_router)

    graph.set_entry_point("agent1")
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "agent3")
    graph.add_edge("agent3", "agent4")
    graph.add_edge("agent4", END)

    return graph.compile()


pipeline = build_pipeline()


def run_pipeline(quote_dict: dict) -> dict:
    """Main entry point called by FastAPI."""
    state = QuoteState(
        quote_id            = quote_dict.get("Quote_Num", "UNKNOWN"),
        agent_type          = quote_dict.get("Agent_Type", "EA"),
        region              = quote_dict.get("Region", "A"),
        policy_type         = quote_dict.get("Policy_Type", "Car"),
        hh_vehicles         = int(quote_dict.get("HH_Vehicles", 1)),
        hh_drivers          = int(quote_dict.get("HH_Drivers", 1)),
        driver_age          = int(quote_dict.get("Driver_Age", 35)),
        driving_exp         = int(quote_dict.get("Driving_Exp", 10)),
        prev_accidents      = int(quote_dict.get("Prev_Accidents", 0)),
        prev_citations      = int(quote_dict.get("Prev_Citations", 0)),
        gender              = quote_dict.get("Gender", "Male"),
        marital_status      = quote_dict.get("Marital_Status", "Single"),
        education           = quote_dict.get("Education", "Bachelors"),
        sal_range           = quote_dict.get("Sal_Range", "> $ 25 K <= $ 40 K"),
        coverage            = quote_dict.get("Coverage", "Balanced"),
        veh_usage           = quote_dict.get("Veh_Usage", "Commute"),
        annual_miles_range  = quote_dict.get("Annual_Miles_Range", "> 7.5 K & <= 15 K"),
        vehicl_cost_range   = quote_dict.get("Vehicl_Cost_Range", "> $ 10 K <= $ 20 K"),
        re_quote            = quote_dict.get("Re_Quote", "No"),
        quoted_premium      = float(quote_dict.get("Quoted_Premium", 700)),
        q_valid_dt          = quote_dict.get("Q_Valid_DT", "2024-12-31"),
        # Agent outputs — start as None
        risk_tier=None, risk_score=None, risk_shap=None, risk_shap_chart=None,
        bind_score=None, bind_pct=None, conv_shap=None, conv_shap_chart=None,
        premium_flag=None, premium_ratio=None, llm_reasoning=None,
        final_decision=None, escalation_reason=None, llm_summary=None, audit_trail=None,
    )

    result = pipeline.invoke(state)
    return dict(result)