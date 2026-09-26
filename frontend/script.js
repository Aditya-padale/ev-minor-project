const API_BASE = (() => {
  if (window.location.protocol === "file:") {
    return "http://localhost:5000";
  }
  return "";
})();

const el = (id) => document.getElementById(id);

const vehicleTypeSel = el("vehicleType");
const chargerPresetSel = el("chargerPreset");
const chargerKwInput = el("chargerKw");
const ambientTemp = el("ambientTemp");
const tempReadout = el("tempReadout");
const form = el("predictForm");
const errorMsg = el("errorMsg");

const resultEmpty = el("resultEmpty");
const resultContent = el("resultContent");

let CHARGER_PRESETS = {};
let lastPayload = null;
let lastResult = null;

// ---------------------------------------------------------------- init ---
async function init() {
  try {
    const res = await fetch(`${API_BASE}/api/meta`);
    if (!res.ok) throw new Error("meta fetch failed");
    const meta = await res.json();

    meta.vehicle_types.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      vehicleTypeSel.appendChild(opt);
    });
    vehicleTypeSel.value = "Sedan";

    CHARGER_PRESETS = meta.charger_presets;
    Object.keys(CHARGER_PRESETS).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = `${name}`;
      chargerPresetSel.appendChild(opt);
    });
    const customOpt = document.createElement("option");
    customOpt.value = "__custom__";
    customOpt.textContent = "Custom power…";
    chargerPresetSel.appendChild(customOpt);
    chargerPresetSel.value = "DC Fast (30 kW)";
    chargerKwInput.value = CHARGER_PRESETS["DC Fast (30 kW)"];

    setStatus(true);
  } catch (e) {
    setStatus(false);
  }
}

function setStatus(ok) {
  const dot = el("statusDot");
  const text = el("statusText");
  if (ok) {
    dot.classList.remove("offline");
    text.textContent = "model online";
  } else {
    dot.classList.add("offline");
    text.textContent = "backend offline — run app.py";
  }
}

// ------------------------------------------------------------ handlers ---
chargerPresetSel.addEventListener("change", () => {
  const val = chargerPresetSel.value;
  if (val !== "__custom__" && CHARGER_PRESETS[val] !== undefined) {
    chargerKwInput.value = CHARGER_PRESETS[val];
  }
});

ambientTemp.addEventListener("input", () => {
  tempReadout.textContent = `${ambientTemp.value}\u00b0C`;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorMsg.textContent = "";

  const payload = {
    vehicle_type: vehicleTypeSel.value,
    battery_capacity_kwh: parseFloat(el("batteryCapacity").value),
    start_soc_percent: parseFloat(el("startSoc").value),
    target_soc_percent: parseFloat(el("targetSoc").value),
    charger_power_kw: parseFloat(chargerKwInput.value),
    ambient_temp_c: parseFloat(ambientTemp.value),
    battery_health_percent: parseFloat(el("batteryHealth").value),
    departure_hours: parseFloat(el("departureHours").value),
    electricity_rate_per_kwh: parseFloat(el("electricityRate").value),
    carbon_intensity_g_per_kwh: parseFloat(el("carbonIntensity").value),
    battery_preconditioned: el("preconditioned").checked,
  };

  if (payload.start_soc_percent >= payload.target_soc_percent) {
    errorMsg.textContent = "Target charge must be higher than current charge.";
    return;
  }

  const btn = form.querySelector(".predict-btn");
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      errorMsg.textContent = data.error || "Prediction failed.";
      return;
    }
    renderResult(data, payload);
  } catch (err) {
    errorMsg.textContent = "Could not reach backend. Is app.py running?";
  } finally {
    btn.disabled = false;
  }
});

// -------------------------------------------------------------- render ---
function renderResult(data, payload) {
  lastPayload = payload;
  lastResult = data;
  resultEmpty.hidden = true;
  resultContent.hidden = false;

  el("heroTime").textContent = data.predicted_hours_minutes;
  el("heroSub").textContent =
    `${payload.vehicle_type} · ${payload.start_soc_percent}% \u2192 ${payload.target_soc_percent}% · ${payload.charger_power_kw} kW`;

  el("statEnergy").textContent = `${data.energy_needed_kwh} kWh`;
  el("statPower").textContent = `${data.avg_effective_power_kw} kW`;
  el("statModel").textContent = data.model_used;
  el("statRange").textContent = data.plan.confidence_range_minutes.label;
  el("statCost").textContent = `${_formatCost(data.plan.estimated_cost)} / session`;
  el("summaryWindow").textContent = data.plan.confidence_range_minutes.label;
  el("summaryEnergy").textContent = `${data.energy_needed_kwh} kWh`;
  el("summaryCost").textContent = _formatCost(data.plan.estimated_cost);
  el("summaryCarbon").textContent = `${data.plan.estimated_carbon_kg} kg`;
  el("summaryDeadline").textContent = data.plan.deadline_status;
  el("planStrategy").textContent = data.plan.strategy;
  el("planDetail").textContent = data.plan.strategy_detail;
  el("planMeta").textContent = `${data.plan.estimated_carbon_kg} kg CO2e · ${data.plan.battery_health_advice}`;

  const stat80Card = el("stat80Card");
  if (data.time_to_80_percent_minutes) {
    stat80Card.hidden = false;
    el("stat80").textContent = `${data.time_to_80_percent_minutes} min`;
  } else {
    stat80Card.hidden = true;
  }

  const fillPct = payload.target_soc_percent;
  el("progressFill").style.width = `${fillPct}%`;
  el("markerStart").style.left = `${payload.start_soc_percent}%`;
  el("markerTarget").style.left = `${payload.target_soc_percent}%`;
  el("progressStartLabel").textContent = `${payload.start_soc_percent}%`;
  el("progressTargetLabel").textContent = `${payload.target_soc_percent}%`;

  const suggBox = el("suggestions");
  suggBox.innerHTML = "";
  data.suggestions.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "suggestion-card";
    card.style.animationDelay = `${i * 60}ms`;
    card.innerHTML = `<div class="s-title">${escapeHtml(s.title)}</div><div class="s-detail">${escapeHtml(s.detail)}</div>`;
    suggBox.appendChild(card);
  });

  renderCurve(data.curve);
  renderCare(data.battery_care);
  loadPlanningViews(payload);
}

function renderCurve(curve) {
  const width = 640;
  const height = 230;
  const pad = { left: 42, right: 18, top: 18, bottom: 30 };
  const maxTime = Math.max(curve.completion_minutes, 1);
  const maxSoc = Math.max(curve.target_soc_percent, 1);
  const x = (value) => pad.left + (value / maxTime) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - ((value / maxSoc) * (height - pad.top - pad.bottom));
  const line = curve.points.map((point) => `${x(point.minutes).toFixed(1)},${y(point.soc_percent).toFixed(1)}`).join(" ");
  const ccPoints = curve.points.filter((point) => point.phase === "CC").map((point) => `${x(point.minutes).toFixed(1)},${y(point.soc_percent).toFixed(1)}`).join(" ");
  const cvPoints = curve.points.filter((point) => point.phase === "CV").map((point) => `${x(point.minutes).toFixed(1)},${y(point.soc_percent).toFixed(1)}`).join(" ");
  const marker = curve.soc_80_percent ? `<line class="chart-marker" x1="${x(curve.cc_minutes)}" y1="${pad.top}" x2="${x(curve.cc_minutes)}" y2="${height - pad.bottom}" /><text x="${x(curve.cc_minutes) + 5}" y="${pad.top + 12}">80% / CV</text>` : "";
  el("curveChart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Charging curve showing constant current and tapering constant voltage phases"><line class="chart-axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" /><line class="chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" /><polyline class="curve-underlay" points="${line}" /><polyline class="curve-line cc-line" points="${ccPoints}" /><polyline class="curve-line cv-line" points="${cvPoints}" /><circle class="chart-point" cx="${x(0)}" cy="${y(curve.current_soc_percent)}" r="4" /><circle class="chart-point target-point" cx="${x(curve.completion_minutes)}" cy="${y(curve.target_soc_percent)}" r="5" />${marker}<text class="axis-label" x="${pad.left}" y="${height - 8}">0 min</text><text class="axis-label" x="${width - pad.right - 32}" y="${height - 8}">${curve.completion_minutes} min</text><text class="axis-label" x="8" y="${height - pad.bottom + 4}">0%</text><text class="axis-label" x="8" y="${pad.top + 4}">${curve.target_soc_percent}%</text></svg>`;
  el("curveComplete").textContent = `Finish · ${_formatMinutes(curve.completion_minutes)}`;
}

function renderCare(care) {
  el("careLevel").textContent = `${care.stress_level} stress · ${care.stress_score}/4`;
  el("careLevel").className = `care-level ${care.stress_level.toLowerCase()}`;
  el("careAdvice").textContent = care.advice;
  el("careFactors").textContent = `Factors: ${care.factors.join(" · ")}`;
}

async function loadPlanningViews(payload) {
  try {
    const [recommendation, mode] = await Promise.all([
      apiPost("/api/charger-recommendation", payload),
      apiPost("/api/charging-plan", { ...payload, mode: document.querySelector(".mode-tab.active").dataset.mode }),
    ]);
    renderRecommendation(recommendation);
    renderMode(mode);
  } catch (error) {
    el("recommendation").textContent = error.message;
  }
}

function apiPost(path, body) {
  return fetch(`${API_BASE}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.error || "Request failed."); return data; });
}

function renderRecommendation(data) {
  const chosen = data.recommended;
  el("recommendation").innerHTML = `<strong>RECOMMENDED CHARGER<br><b>${chosen.charger_power_kw} kW</b></strong><span>Estimated completion: ${escapeHtml(chosen.charging_time)}<br>Deadline: <em class="${chosen.deadline_status === "SAFE" ? "on-time" : "risk"}">${chosen.deadline_status}</em> · Safety buffer: ${_formatMinutes(Math.max(chosen.safety_buffer_minutes, 0))}</span>`;
  el("chargerTable").innerHTML = `<div class="charger-row charger-head"><span>Power</span><span>Time</span><span>Cost</span><span>CO2</span><span>Status</span></div>${data.options.map((option) => `<div class="charger-row"><span>${option.charger_power_kw} kW</span><span>${escapeHtml(option.charging_time)}</span><span>${_formatCost(option.estimated_cost)}</span><span>${option.estimated_carbon_kg} kg</span><span class="${option.deadline_status === "SAFE" ? "on-time" : "risk"}">${option.deadline_status}</span></div>`).join("")}`;
}

function renderMode(data) {
  const selected = data.selected;
  el("modeResult").innerHTML = `<strong>${data.mode}</strong> selects <b>${selected.charger_power_kw} kW</b> · ${escapeHtml(selected.charging_time)} · ${_formatCost(selected.estimated_cost)} · ${selected.estimated_carbon_kg} kg CO2e`;
  el("comparisonBars").innerHTML = data.options.map((option) => `<div class="bar-row"><span>${option.charger_power_kw} kW</span><div><i style="width:${Math.min(option.charging_time_minutes / Math.max(...data.options.map((item) => item.charging_time_minutes)) * 100, 100)}%"></i></div><small>${option.charging_time} · ${_formatCost(option.estimated_cost)} · ${option.estimated_carbon_kg} kg CO2</small></div>`).join("");
}

function _formatMinutes(minutes) {
  const rounded = Math.max(0, Math.round(minutes));
  return rounded >= 60 ? `${Math.floor(rounded / 60)} hr ${rounded % 60} min` : `${rounded} min`;
}

function _formatCost(cost) {
  return `₹${Number(cost).toFixed(2)}`;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

init();

document.querySelectorAll(".mode-tab").forEach((button) => button.addEventListener("click", async () => {
  document.querySelectorAll(".mode-tab").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  if (lastPayload) renderMode(await apiPost("/api/charging-plan", { ...lastPayload, mode: button.dataset.mode }));
}));

el("simulateBtn").addEventListener("click", async () => {
  if (!lastPayload) return;
  const scenario = {
    ambient_temp_c: Number(el("simTemp").value),
    charger_power_kw: Number(el("simPower").value),
    battery_health_percent: Number(el("simHealth").value),
    target_soc_percent: Number(el("simTarget").value),
    departure_hours: Number(el("simDeparture").value),
  };
  try {
    const data = await apiPost("/api/simulate", { ...lastPayload, scenario });
    el("simulationResult").innerHTML = `<div class="simulation-table"><div class="sim-row sim-head"><span>Parameter</span><span>Current</span><span>Scenario</span></div>${data.comparison.map((item) => `<div class="sim-row"><span>${item.parameter}</span><span>${item.current}${item.unit ? ` ${item.unit}` : ""}</span><span>${item.scenario}${item.unit ? ` ${item.unit}` : ""}</span></div>`).join("")}</div>`;
  } catch (error) { el("simulationResult").textContent = error.message; }
});

el("scheduleBtn").addEventListener("click", async () => {
  const lines = el("fleetInput").value.split("\n").map((line) => line.trim()).filter(Boolean);
  const vehicles = lines.map((line) => {
    const [name, energy, deadline, power] = line.split(",").map((value) => value.trim());
    return { name, energy_needed_kwh: Number(energy), deadline_hours: Number(deadline), charger_power_kw: Number(power) };
  });
  const result = el("fleetResult");
  try {
    const res = await fetch(`${API_BASE}/api/fleet-schedule`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vehicles, site_limit_kw: Number(el("siteLimit").value) }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not build schedule.");
    result.hidden = false;
    result.innerHTML = `<div class="site-meter"><strong>SITE LIMIT: ${data.site_limit_kw} kW</strong><span>USED: ${data.used_power_kw} kW · AVAILABLE: ${data.available_power_kw} kW · ${data.utilization_percent}% utilized</span><i><b style="width:${data.utilization_percent}%"></b></i></div><div class="fleet-table"><div class="fleet-row fleet-head"><span>Vehicle</span><span>Energy</span><span>Power</span><span>Duration</span><span>Deadline</span><span>Status</span></div>${data.schedule.map((item) => `<div class="fleet-row"><span>${escapeHtml(item.name)}</span><span>${item.energy_needed_kwh} kWh</span><span>${item.assigned_power_kw} kW</span><span>${item.estimated_duration_hours ?? "—"} hr</span><span>${item.deadline_hours} hr</span><span class="${item.status === "at risk" ? "risk" : "on-time"}">${item.status}</span></div>`).join("")}</div>`;
  } catch (error) {
    result.hidden = false;
    result.textContent = error.message;
  }
});
