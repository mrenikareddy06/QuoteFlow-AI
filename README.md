# QuoteFlow AI 🤖
### Autonomous Quote Agent Pipeline

> A multi-agent AI system that autonomously processes auto insurance quotes end-to-end — profiling risk, predicting conversion, advising on premiums, and routing decisions without human intervention.

![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-blue?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Conversion-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Risk-orange?style=for-the-badge)
![Llama 3.1](https://img.shields.io/badge/Llama_3.1-Local_LLM-purple?style=for-the-badge)

---

## Problem Statement

Auto insurance carriers generate thousands of quotes daily. Yet **only 1 in 5 quotes converts to a bound policy**. The rest expire silently — representing millions in wasted acquisition spend. Every unconverted quote currently demands a human to investigate manually.

**QuoteFlow AI eliminates that bottleneck.**

---

## The 4-Agent Pipeline

| Agent | Model | Mode | Role |
|---|---|---|---|
| **01 — Risk Profiler** | XGBoost 2.0 | Fully Automatic | Analyses accident history, citations, driving experience, age, vehicle usage → outputs LOW / MEDIUM / HIGH risk tier with SHAP explanation |
| **02 — Conversion Predictor** | LightGBM + SMOTE | Fully Automatic | Scores each unbound quote 0–100% bind probability. 3-layer imbalance fix handles the 22% class skew |
| **03 — Premium Advisor** | Llama 3.1 8B (Ollama) | Hybrid | Reasons whether quoted premium is a conversion blocker by comparing it to the customer's salary bracket using chain-of-thought LLM reasoning. No API key required |
| **04 — Decision Router** | Rules + Llama 3.1 | Escalate-Only | Combines all upstream outputs → routes to Auto-Approve (~45%), Follow-Up (~35%), or Escalate (~20%). Only escalations ever touch a human underwriter |

---

## Tech Stack
```
LangGraph        → Agent orchestration & stateful pipeline
XGBoost          → Agent 1: Risk profiling
LightGBM + SMOTE → Agent 2: Conversion prediction (22% imbalance handled)
Llama 3.1 8B     → Agent 3 & 4: Local LLM reasoning (via Ollama)
SHAP             → Explainability at every agent node
FastAPI          → Backend REST API
Vanilla JS       → Frontend dashboard
Chart.js         → Live updating charts
```

---

## How to Run

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com) installed
- VS Code with **Live Server** extension

### Step 1 — Create Virtual Environment
```bash
python -m venv venv
```

### Step 2 — Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Start Llama 3.1 (Local LLM)
```bash
ollama run llama3.1
```
>  Keep this terminal open. First run downloads the model (~4.7GB). Subsequent runs are instant.

### Step 5 — Start Backend (New Terminal)
```bash
cd backend
uvicorn main:app --reload --port 8000
```
> Backend runs at `http://127.0.0.1:8000`

### Step 6 — Start Frontend
> Right-click `frontend/index.html` → **Open with Live Server**

> Frontend runs at `http://127.0.0.1:5500/frontend/index.html`

---

##  Project Structure
```
QuoteFlow_AI/
├── backend/
│   ├── data/
│   │   └── quotes.csv          ← 146,259 real insurance quotes
│   ├── models/
│   │   ├── xgb_risk.pkl        ← Trained XGBoost risk model
│   │   ├── lgb_conv.pkl        ← Trained LightGBM conversion model
│   │   ├── encoders.pkl        ← Feature encoders
│   │   ├── conv_features.pkl   ← Conversion feature list
│   │   └── risk_features.pkl   ← Risk feature list
│   ├── main.py                 ← FastAPI endpoints
│   ├── pipeline.py             ← LangGraph 4-agent pipeline
│   ├── explainer.py            ← SHAP explainability
│   ├── train_models.py         ← Model training script
│   └── requirements.txt
├── frontend/
│   ├── index.html              ← Main dashboard
│   ├── app.js                  ← Dashboard logic
│   └── style.css               ← Styling
└── README.md
```

---

##  Escalation Triggers

A quote escalates when **any one** condition is met:
```
- Bind score < 35%
- Risk = HIGH and Bind score < 50%  
- Premium BLOCKER + Re-Quote + HIGH risk
- Model confidence < 60%
```

---

## Dataset

- **146,259** real auto insurance quotes
- **22.2%** conversion rate (1 in 5 quotes binds)
- **8 regions** (A through H), EA and IA agent types
- 3-layer imbalance fix: SMOTE + class_weight='balanced' + threshold tuning

---

## Built For

**GITAM AI DAY Hackathon** — Use Case 3: Autonomous Quote Agents  
