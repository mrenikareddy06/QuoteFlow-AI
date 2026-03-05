"""
main.py
FastAPI backend — run with:
    uvicorn main:app --reload --port 8000
"""

import pandas as pd
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pipeline import run_pipeline

app = FastAPI(title="QuoteFlow AI", version="1.0.0")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CSV for demo feed
try:
    df = pd.read_csv("data/quotes.csv")
    print(f"Loaded {len(df)} quotes from CSV")
except Exception as e:
    df = None
    print(f"CSV not found: {e}")

# In-memory store of processed quotes
processed_quotes = []
escalation_queue = []


# ── HEALTH CHECK ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "QuoteFlow AI is running", "version": "1.0.0"}


# ══════════════════════════════════════════════════════════════
# MAIN ENDPOINT — process one quote through all 4 agents
# ══════════════════════════════════════════════════════════════
@app.post("/process-quote")
async def process_quote(quote: dict):
    """
    Takes a quote dict with CSV columns.
    Runs through LangGraph 4-agent pipeline.
    Returns full result with SHAP charts as base64.
    """
    result = run_pipeline(quote)

    # Store for dashboard
    summary = {
        "quote_id":        result.get("quote_id"),
        "region":          result.get("region"),
        "agent_type":      result.get("agent_type"),
        "risk_tier":       result.get("risk_tier"),
        "bind_pct":        result.get("bind_pct"),
        "premium_flag":    result.get("premium_flag"),
        "final_decision":  result.get("final_decision"),
    }
    processed_quotes.append(summary)
    if result.get("final_decision") == "ESCALATE":
        escalation_queue.append({
            **summary,
            "escalation_reason": result.get("escalation_reason"),
            "llm_summary":       result.get("llm_summary"),
        })

    return result


# ══════════════════════════════════════════════════════════════
# DASHBOARD STATS
# ══════════════════════════════════════════════════════════════
@app.get("/stats")
def get_stats():
    """Aggregate stats for the React dashboard."""
    total = len(processed_quotes)
    if total == 0:
        return {
            "total": 0,
            "risk_distribution":     {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "routing_distribution":  {"AUTO_APPROVE": 0, "FOLLOW_UP": 0, "ESCALATE": 0},
            "avg_bind_score":        0,
            "recent_quotes":         [],
            "escalation_count":      0,
        }

    risk_dist  = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    route_dist = {"AUTO_APPROVE": 0, "FOLLOW_UP": 0, "ESCALATE": 0}
    bind_sum   = 0

    for q in processed_quotes:
        r = q.get("risk_tier", "LOW")
        if r in risk_dist:
            risk_dist[r] += 1
        d = q.get("final_decision", "FOLLOW_UP")
        if d in route_dist:
            route_dist[d] += 1
        bind_sum += q.get("bind_pct", 0)

    return {
        "total":                 total,
        "risk_distribution":     risk_dist,
        "routing_distribution":  route_dist,
        "avg_bind_score":        round(bind_sum / total, 1),
        "recent_quotes":         processed_quotes[-20:][::-1],
        "escalation_count":      len(escalation_queue),
    }


# ══════════════════════════════════════════════════════════════
# ESCALATION QUEUE
# ══════════════════════════════════════════════════════════════
@app.get("/escalation")
def get_escalation():
    """Returns list of quotes that need human review."""
    return {"escalations": escalation_queue[-50:][::-1]}


# ══════════════════════════════════════════════════════════════
# DEMO — run a random quote from CSV
# ══════════════════════════════════════════════════════════════
@app.get("/demo-quote")
def demo_quote():
    if df is None:
        return {"error": "CSV not loaded"}
    row = df.sample(1).iloc[0].to_dict()
    clean = {}
    for k, v in row.items():
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                clean[k] = ""
            else:
                clean[k] = str(v)
        except:
            clean[k] = str(v)
    return clean
# ══════════════════════════════════════════════════════════════
# BATCH DEMO — process N random quotes
# ══════════════════════════════════════════════════════════════
@app.post("/batch-demo")
async def batch_demo(body: dict):
    """Process n random quotes for dashboard demo."""
    n = min(int(body.get("n", 5)), 20)
    if df is None:
        return {"error": "CSV not loaded"}
    results = []
    for _, row in df.sample(n).iterrows():
        quote = {k: (int(v) if str(type(v)) == "<class 'numpy.int64'>" else
                     float(v) if str(type(v)) == "<class 'numpy.float64'>" else str(v))
                 for k, v in row.items()}
        result = run_pipeline(quote)
        summary = {
            "quote_id":       result.get("quote_id"),
            "risk_tier":      result.get("risk_tier"),
            "bind_pct":       result.get("bind_pct"),
            "premium_flag":   result.get("premium_flag"),
            "final_decision": result.get("final_decision"),
        }
        processed_quotes.append(summary)
        if result.get("final_decision") == "ESCALATE":
            escalation_queue.append({
                **summary,
                "escalation_reason": result.get("escalation_reason"),
                "llm_summary":       result.get("llm_summary"),
            })
        results.append(summary)
    return {"processed": n, "results": results}