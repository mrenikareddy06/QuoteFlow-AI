// ═══════════════════════════════════════════════════════════
// QuoteFlow AI — app.js
// All frontend logic: API calls, charts, tabs, results
// ═══════════════════════════════════════════════════════════

const API = "http://localhost:8000";

// ── Chart instances ──────────────────────────────────────────
let riskChart    = null;
let routingChart = null;
let bindChart    = null;

// ── Recent bind scores for histogram ────────────────────────
let recentBindScores = [];

// ════════════════════════════════════════════════════════════
// TABS
// ════════════════════════════════════════════════════════════
function showTab(name) {
  document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  event.target.classList.add("active");
  if (name === "dashboard") refreshDashboard();
  if (name === "escalation") loadEscalation();
}

// ════════════════════════════════════════════════════════════
// INIT CHARTS
// ════════════════════════════════════════════════════════════
function initCharts() {
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { labels: { color: "#64748b", font: { size: 11 } } } }
  };

  // Risk donut
  riskChart = new Chart(document.getElementById("riskChart"), {
    type: "doughnut",
    data: {
      labels: ["LOW", "MEDIUM", "HIGH"],
      datasets: [{
        data: [0, 0, 0],
        backgroundColor: ["rgba(16,185,129,0.8)", "rgba(245,158,11,0.8)", "rgba(239,68,68,0.8)"],
        borderColor: ["#10b981", "#f59e0b", "#ef4444"],
        borderWidth: 2,
      }]
    },
    options: {
      ...chartDefaults,
      cutout: "65%",
    }
  });

  // Routing donut
  routingChart = new Chart(document.getElementById("routingChart"), {
    type: "doughnut",
    data: {
      labels: ["AUTO APPROVE", "FOLLOW UP", "ESCALATE"],
      datasets: [{
        data: [0, 0, 0],
        backgroundColor: ["rgba(16,185,129,0.8)", "rgba(245,158,11,0.8)", "rgba(239,68,68,0.8)"],
        borderColor: ["#10b981", "#f59e0b", "#ef4444"],
        borderWidth: 2,
      }]
    },
    options: { ...chartDefaults, cutout: "65%" }
  });

  // Bind score bar chart
  bindChart = new Chart(document.getElementById("bindChart"), {
    type: "bar",
    data: {
      labels: ["0-10", "10-20", "20-30", "30-40", "40-50",
               "50-60", "60-70", "70-80", "80-90", "90-100"],
      datasets: [{
        label: "Quotes",
        data: new Array(10).fill(0),
        backgroundColor: "rgba(0,229,255,0.5)",
        borderColor: "#00e5ff",
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      ...chartDefaults,
      scales: {
        x: { ticks: { color: "#64748b" }, grid: { color: "#1e2940" } },
        y: { ticks: { color: "#64748b" }, grid: { color: "#1e2940" } }
      }
    }
  });
}

// ════════════════════════════════════════════════════════════
// DASHBOARD REFRESH
// ════════════════════════════════════════════════════════════
async function refreshDashboard() {
  try {
    const res  = await fetch(`${API}/stats`);
    const data = await res.json();

    // Header stats
    document.getElementById("hdr-total").textContent = data.total;
    document.getElementById("hdr-bind").textContent  = data.avg_bind_score + "%";
    document.getElementById("hdr-esc").textContent   = data.escalation_count;

    // Update risk chart
    const rd = data.risk_distribution;
    riskChart.data.datasets[0].data = [rd.LOW || 0, rd.MEDIUM || 0, rd.HIGH || 0];
    riskChart.update();

    // Update routing chart
    const rout = data.routing_distribution;
    routingChart.data.datasets[0].data = [
      rout.AUTO_APPROVE || 0,
      rout.FOLLOW_UP    || 0,
      rout.ESCALATE     || 0,
    ];
    routingChart.update();

    // Update bind histogram from recent quotes
    const buckets = new Array(10).fill(0);
    (data.recent_quotes || []).forEach(q => {
      const idx = Math.min(Math.floor((q.bind_pct || 0) / 10), 9);
      buckets[idx]++;
    });
    bindChart.data.datasets[0].data = buckets;
    bindChart.update();

    // Feed table
    renderFeedTable(data.recent_quotes || []);

  } catch (e) {
    console.error("Dashboard refresh failed:", e);
  }
}

// ════════════════════════════════════════════════════════════
// FEED TABLE
// ════════════════════════════════════════════════════════════
function renderFeedTable(quotes) {
  const tbody = document.getElementById("feed-body");
  if (!quotes.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-msg">Run a demo to see quotes appear here</td></tr>`;
    return;
  }

  tbody.innerHTML = quotes.map(q => `
    <tr>
      <td><code>${q.quote_id || "—"}</code></td>
      <td>${q.region || "—"}</td>
      <td>${q.agent_type || "—"}</td>
      <td>${riskBadge(q.risk_tier)}</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="flex:1;height:6px;background:#1e2940;border-radius:3px;overflow:hidden">
            <div style="width:${q.bind_pct||0}%;height:100%;background:#00e5ff;border-radius:3px"></div>
          </div>
          <span style="font-family:monospace;font-size:12px;color:#00e5ff">${q.bind_pct||0}%</span>
        </div>
      </td>
      <td>${flagBadge(q.premium_flag)}</td>
      <td>${decisionBadge(q.final_decision)}</td>
    </tr>
  `).join("");
}

// ════════════════════════════════════════════════════════════
// PROCESS QUOTE
// ════════════════════════════════════════════════════════════
async function submitQuote() {
  const quote = buildQuoteFromForm();
  await processAndShow(quote);
}

function buildQuoteFromForm() {
  return {
    Quote_Num:          document.getElementById("f-quote-id").value,
    Agent_Type:         document.getElementById("f-agent-type").value,
    Region:             document.getElementById("f-region").value,
    Policy_Type:        document.getElementById("f-policy-type").value,
    Driver_Age:         document.getElementById("f-driver-age").value,
    Driving_Exp:        document.getElementById("f-driving-exp").value,
    Prev_Accidents:     document.getElementById("f-prev-acc").value,
    Prev_Citations:     document.getElementById("f-prev-cit").value,
    HH_Vehicles:        document.getElementById("f-hh-veh").value,
    HH_Drivers:         document.getElementById("f-hh-drv").value,
    Sal_Range:          document.getElementById("f-sal").value,
    Coverage:           document.getElementById("f-coverage").value,
    Veh_Usage:          document.getElementById("f-veh-usage").value,
    Annual_Miles_Range: document.getElementById("f-miles").value,
    Vehicl_Cost_Range:  document.getElementById("f-vcost").value,
    Re_Quote:           document.getElementById("f-requote").value,
    Quoted_Premium:     document.getElementById("f-premium").value,
    Gender:             document.getElementById("f-gender").value,
    Marital_Status:     "Single",
    Education:          "Bachelors",
    Q_Valid_DT:         "2025-12-31",
  };
}

async function loadDemo() {
  try {
    const res   = await fetch(`${API}/demo-quote`);
    const quote = await res.json();
    await processAndShow(quote);
  } catch(e) {
    alert("Could not load demo quote. Is the backend running?");
  }
}

async function processAndShow(quote) {
  // Show loading
  document.getElementById("result-panel").classList.add("hidden");
  document.getElementById("loading-panel").classList.remove("hidden");

  // Animate steps
  const steps = ["step1","step2","step3","step4"];
  steps.forEach(s => document.getElementById(s).className = "agent-step");

  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx > 0) document.getElementById(steps[stepIdx-1]).className = "agent-step done";
    if (stepIdx < steps.length) {
      document.getElementById(steps[stepIdx]).className = "agent-step active";
      stepIdx++;
    } else {
      clearInterval(stepInterval);
    }
  }, 600);

  try {
    const res    = await fetch(`${API}/process-quote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(quote),
    });
    const result = await res.json();

    clearInterval(stepInterval);
    steps.forEach(s => document.getElementById(s).className = "agent-step done");

    setTimeout(() => {
      document.getElementById("loading-panel").classList.add("hidden");
      renderResult(result);
      refreshDashboard();
    }, 400);

  } catch(e) {
    clearInterval(stepInterval);
    document.getElementById("loading-panel").classList.add("hidden");
    alert("Pipeline error. Make sure backend is running on port 8000.");
  }
}

function renderResult(r) {
  const panel = document.getElementById("result-panel");
  panel.classList.remove("hidden");

  // Decision banner
  const banner = document.getElementById("decision-banner");
  const decision = r.final_decision || "UNKNOWN";
  const bannerClasses = {
    "AUTO_APPROVE": "banner-auto",
    "FOLLOW_UP":    "banner-follow",
    "ESCALATE":     "banner-esc",
  };
  banner.className = "decision-banner " + (bannerClasses[decision] || "");
  banner.textContent = decision.replace("_", " ");

  // Scores
  document.getElementById("res-risk").textContent  = r.risk_tier  || "—";
  document.getElementById("res-bind").textContent  = (r.bind_pct || 0) + "%";
  document.getElementById("res-flag").textContent  = r.premium_flag || "—";
  document.getElementById("res-ratio").textContent = r.premium_ratio ? r.premium_ratio + "%" : "—";

  // Color scores
  const riskEl = document.getElementById("res-risk");
  riskEl.style.color = r.risk_tier === "HIGH" ? "#ef4444" :
                       r.risk_tier === "MEDIUM" ? "#f59e0b" : "#10b981";

  const flagEl = document.getElementById("res-flag");
  flagEl.style.color = r.premium_flag === "BLOCKER" ? "#ef4444" : "#10b981";

  // Bind bar
  const pct = r.bind_pct || 0;
  document.getElementById("bind-bar-fill").style.width = pct + "%";
  document.getElementById("bind-bar-pct").textContent  = pct + "%";
  document.getElementById("bind-bar-fill").style.background =
    pct >= 65 ? "#10b981" : pct >= 35 ? "#f59e0b" : "#ef4444";

  // Reasoning
  document.getElementById("res-reasoning").textContent = r.llm_reasoning || "Not available";

  // Escalation summary
  const sumWrap = document.getElementById("summary-wrap");
  if (decision === "ESCALATE" && r.llm_summary) {
    sumWrap.classList.remove("hidden");
    document.getElementById("res-summary").textContent = r.llm_summary;
  } else {
    sumWrap.classList.add("hidden");
  }

  // SHAP charts
  if (r.risk_shap_chart) {
    const img = document.getElementById("risk-shap-img");
    img.src = "data:image/png;base64," + r.risk_shap_chart;
    img.classList.remove("hidden");
  }
  if (r.conv_shap_chart) {
    const img = document.getElementById("conv-shap-img");
    img.src = "data:image/png;base64," + r.conv_shap_chart;
    img.classList.remove("hidden");
  }

  // Audit trail
  const audit = r.audit_trail || [];
  document.getElementById("audit-trail").innerHTML = audit.map(a => `
    <div class="audit-row">
      <span class="audit-agent">${a.agent}</span>
      <span class="audit-output">${a.output}</span>
      <span class="audit-model">${a.model}</span>
    </div>
  `).join("");
}

// ════════════════════════════════════════════════════════════
// ESCALATION QUEUE
// ════════════════════════════════════════════════════════════
async function loadEscalation() {
  try {
    const res  = await fetch(`${API}/escalation`);
    const data = await res.json();
    const list = data.escalations || [];
    const tbody = document.getElementById("esc-body");

    if (!list.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="empty-msg">No escalations yet</td></tr>`;
      return;
    }

    tbody.innerHTML = list.map(e => `
      <tr>
        <td><code>${e.quote_id || "—"}</code></td>
        <td>${riskBadge(e.risk_tier)}</td>
        <td><span style="font-family:monospace;color:#00e5ff">${e.bind_pct||0}%</span></td>
        <td>${flagBadge(e.premium_flag)}</td>
        <td style="font-size:12px;color:#94a3b8">${e.escalation_reason || "—"}</td>
        <td style="font-size:12px;color:#64748b;max-width:240px">${e.llm_summary || "—"}</td>
      </tr>
    `).join("");
  } catch(e) {
    console.error("Escalation fetch failed:", e);
  }
}

// ════════════════════════════════════════════════════════════
// BATCH DEMO
// ════════════════════════════════════════════════════════════
async function runBatch(n) {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = "Processing...";
  try {
    await fetch(`${API}/batch-demo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n }),
    });
    await refreshDashboard();
  } catch(e) {
    alert("Batch failed. Is the backend running?");
  }
  btn.disabled = false;
  btn.textContent = `Batch Demo (${n})`;
}

// ════════════════════════════════════════════════════════════
// FILL RANDOM FORM
// ════════════════════════════════════════════════════════════
function fillRandom() {
  const agents   = ["EA", "IA"];
  const regions  = ["A","B","C","D","E","F","G","H"];
  const policies = ["Car","Van","Truck"];
  const coverage = ["Basic","Balanced","Enhanced"];
  const usage    = ["Commute","Pleasure","Business"];
  const sal      = [
    "<= $ 25 K", "> $ 25 K <= $ 40 K",
    "> $ 40 K <= $ 60 K", "> $ 60 K <= $ 90 K", "> $ 90 K"
  ];

  const r = a => a[Math.floor(Math.random() * a.length)];

  document.getElementById("f-quote-id").value    = "AQ-DEMO-" + Math.floor(Math.random()*99999);
  document.getElementById("f-agent-type").value  = r(agents);
  document.getElementById("f-region").value      = r(regions);
  document.getElementById("f-policy-type").value = r(policies);
  document.getElementById("f-driver-age").value  = Math.floor(Math.random()*40) + 18;
  document.getElementById("f-driving-exp").value = Math.floor(Math.random()*30) + 1;
  document.getElementById("f-prev-acc").value    = Math.random() > 0.85 ? "1" : "0";
  document.getElementById("f-prev-cit").value    = Math.random() > 0.85 ? "1" : "0";
  document.getElementById("f-hh-veh").value      = Math.floor(Math.random()*3) + 1;
  document.getElementById("f-hh-drv").value      = Math.floor(Math.random()*3) + 1;
  document.getElementById("f-sal").value         = r(sal);
  document.getElementById("f-coverage").value    = r(coverage);
  document.getElementById("f-veh-usage").value   = r(usage);
  document.getElementById("f-requote").value     = Math.random() > 0.75 ? "Yes" : "No";
  document.getElementById("f-premium").value     =
    (Math.random() * 350 + 600).toFixed(2);
}

// ════════════════════════════════════════════════════════════
// BADGE HELPERS
// ════════════════════════════════════════════════════════════
function riskBadge(tier) {
  const cls = { LOW: "badge-low", MEDIUM: "badge-medium", HIGH: "badge-high" };
  return `<span class="badge ${cls[tier] || ''}">${tier || "—"}</span>`;
}

function flagBadge(flag) {
  const cls = { OK: "badge-ok", BLOCKER: "badge-block" };
  return `<span class="badge ${cls[flag] || ''}">${flag || "—"}</span>`;
}

function decisionBadge(dec) {
  const cls = {
    AUTO_APPROVE: "badge-auto",
    FOLLOW_UP:    "badge-follow",
    ESCALATE:     "badge-esc",
  };
  const label = (dec || "—").replace("_", " ");
  return `<span class="badge ${cls[dec] || ''}">${label}</span>`;
}

// ════════════════════════════════════════════════════════════
// AUTO REFRESH DASHBOARD EVERY 5s
// ════════════════════════════════════════════════════════════
function startAutoRefresh() {
  setInterval(() => {
    const active = document.querySelector(".tab-content.active");
    if (active && active.id === "tab-dashboard") {
      refreshDashboard();
    }
  }, 5000);
}

// ════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  refreshDashboard();
  startAutoRefresh();
});