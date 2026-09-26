"""
app.py
------
Flask API + static file server for the EV Charging Time Prediction
and Optimization app.

Endpoints
---------
GET  /api/health              -> service + model status
GET  /api/meta                -> vehicle types, charger presets, model metrics
POST /api/predict              -> {predicted_minutes, breakdown, suggestions}
POST /api/fleet-schedule       -> deadline-aware multi-vehicle schedule
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model" / "ev_charging_model.pkl"
METRICS_PATH = HERE / "model" / "metrics.json"
FRONTEND_DIR = HERE.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

_bundle = None
_metrics = None


def load_model():
    global _bundle, _metrics
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    else:
        _bundle = None
    if METRICS_PATH.exists():
        _metrics = json.loads(METRICS_PATH.read_text())
    else:
        _metrics = None


load_model()

CHARGER_PRESETS = {
    "Slow AC (3.3 kW)": 3.3,
    "Standard AC (7.2 kW)": 7.2,
    "Fast AC (11 kW)": 11.0,
    "DC Fast (30 kW)": 30.0,
    "DC Ultra-Fast (60 kW)": 60.0,
}

VEHICLE_TYPES = ["Hatchback", "Sedan", "SUV", "Two-Wheeler"]


# ---------------------------------------------------------------------
# Optimization logic (rule-based, layered on top of the ML prediction)
# ---------------------------------------------------------------------
def build_suggestions(payload, predicted_minutes):
    suggestions = []
    temp = payload["ambient_temp_c"]
    target = payload["target_soc_percent"]
    start = payload["start_soc_percent"]
    charger_kw = payload["charger_power_kw"]

    if target > 80:
        suggestions.append({
            "title": "Stop at 80% when possible",
            "detail": (
                "Charging beyond 80% enters the slow CV taper phase, where the "
                "battery management system deliberately throttles current. "
                "Charging only to 80% instead of 100% typically saves a "
                "disproportionate share of total session time."
            ),
        })

    if temp < 10:
        suggestions.append({
            "title": "Cold-weather derating in effect",
            "detail": (
                f"At {temp:.0f}\u00b0C the battery accepts charge more slowly. "
                "If possible, precondition the battery while driving to the "
                "station, or charge during a warmer part of the day."
            ),
        })
    elif temp > 38:
        suggestions.append({
            "title": "High-temperature throttling possible",
            "detail": (
                f"At {temp:.0f}\u00b0C thermal management may reduce charge "
                "current to protect the cells. Charging in shade or during "
                "cooler evening hours can shorten the session."
            ),
        })

    if charger_kw <= 7.2 and (target - start) > 40:
        suggestions.append({
            "title": "Consider a faster charger for this session",
            "detail": (
                "This is a large state-of-charge gain on a lower-power "
                "charger. A DC fast charger (30\u201360 kW) would cut the "
                "predicted time substantially for the 0\u201380% portion."
            ),
        })

    if start > 20 and target - start < 15:
        suggestions.append({
            "title": "Top-up sessions are efficient",
            "detail": (
                "Small top-ups in the mid-SOC range charge at close to peak "
                "power \u2014 this is one of the most time-efficient ways to use "
                "a charging stop."
            ),
        })

    suggestions.append({
        "title": "Avoid grid peak hours",
        "detail": (
            "Public DC chargers are often shared or grid-limited during "
            "evening peak demand (typically 6\u201310 PM). Off-peak sessions "
            "tend to sustain closer to rated power."
        ),
    })

    return suggestions


def build_plan(payload, predicted_minutes):
    """Return practical planning estimates around the ML point prediction."""
    rate = payload["electricity_rate"]
    carbon_intensity = payload["carbon_intensity"]
    battery_health = payload["battery_health"]
    departure_hours = payload["departure_hours"]
    energy_needed = payload["energy_needed_kwh"]

    # The interval deliberately reflects model MAE plus live conditions rather
    # than presenting a false sense of precision.
    uncertainty_factor = 0.9 if payload["battery_preconditioned"] else 1.0
    uncertainty = max(8.0, predicted_minutes * 0.12 * uncertainty_factor)
    low = max(1.0, predicted_minutes - uncertainty)
    high = predicted_minutes + uncertainty
    cost = energy_needed * rate
    carbon_kg = energy_needed * carbon_intensity / 1000.0
    deadline_minutes = departure_hours * 60.0
    buffer_minutes = deadline_minutes - high

    if buffer_minutes >= 20:
        strategy = "Charge during the lowest-cost window"
        strategy_detail = (
            f"You have about {round(buffer_minutes)} minutes of buffer. "
            "A slower off-peak session can finish comfortably while reducing cost and battery stress."
        )
    elif buffer_minutes >= 0:
        strategy = "Start now and monitor the session"
        strategy_detail = (
            "The deadline is achievable, but the uncertainty range leaves little spare time. "
            "Keep battery preconditioning enabled if available."
        )
    else:
        strategy = "Use the fastest available charger"
        strategy_detail = (
            f"The upper estimate misses the deadline by about {round(abs(buffer_minutes))} minutes. "
            "Reduce the target SOC or switch to a higher-power charger if possible."
        )

    if payload["battery_preconditioned"]:
        strategy_detail += " Preconditioning is active, so the timing range is slightly tighter."

    if battery_health < 80:
        health_advice = "Battery health is low; prefer moderate power and an 80% target when the trip allows."
    elif battery_health < 90:
        health_advice = "Use high-power DC charging only when needed and avoid leaving the battery at 100%."
    else:
        health_advice = "Battery health is strong; regular 20-80% sessions will help preserve it."

    return {
        "confidence_range_minutes": {
            "low": round(low, 1),
            "high": round(high, 1),
            "label": f"{_fmt_hm(low)} - {_fmt_hm(high)}",
        },
        "estimated_cost": round(cost, 2),
        "estimated_carbon_kg": round(carbon_kg, 2),
        "battery_health_advice": health_advice,
        "strategy": strategy,
        "strategy_detail": strategy_detail,
    }


def _validate_number(data, key, default, minimum, maximum):
    value = float(data.get(key, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


AVAILABLE_CHARGER_POWERS = [7.2, 11.0, 22.0, 30.0, 60.0]


def _normalized_payload(data, charger_override=None):
    """Validate request data once so every planner uses the same inputs."""
    vehicle_type = str(data.get("vehicle_type", ""))
    capacity = _validate_number(data, "battery_capacity_kwh", None, 0.1, 250)
    start_soc = _validate_number(data, "start_soc_percent", None, 0, 99.9)
    target_soc = _validate_number(data, "target_soc_percent", None, 0.1, 100)
    charger_kw = charger_override if charger_override is not None else _validate_number(
        data, "charger_power_kw", None, 0.5, 350
    )
    temp_c = _validate_number(data, "ambient_temp_c", None, -30, 55)
    battery_health = _validate_number(data, "battery_health_percent", 95, 1, 100)
    departure_hours = _validate_number(data, "departure_hours", 4, 0.25, 72)
    electricity_rate = _validate_number(data, "electricity_rate_per_kwh", 8, 0, 100)
    carbon_intensity = _validate_number(data, "carbon_intensity_g_per_kwh", 650, 0, 1500)

    if vehicle_type not in VEHICLE_TYPES:
        raise ValueError(f"vehicle_type must be one of {VEHICLE_TYPES}")
    if not 0 <= start_soc < target_soc <= 100:
        raise ValueError("Require 0 <= start_soc_percent < target_soc_percent <= 100")

    soc_delta = target_soc - start_soc
    return {
        "vehicle_type": vehicle_type,
        "battery_capacity_kwh": capacity,
        "start_soc_percent": start_soc,
        "target_soc_percent": target_soc,
        "charger_power_kw": float(charger_kw),
        "ambient_temp_c": temp_c,
        "battery_health": battery_health,
        "departure_hours": departure_hours,
        "electricity_rate": electricity_rate,
        "carbon_intensity": carbon_intensity,
        "battery_preconditioned": bool(data.get("battery_preconditioned", False)),
        "energy_needed_kwh": capacity * soc_delta / 100.0,
    }


def _model_minutes(payload):
    energy = payload["energy_needed_kwh"]
    row = pd.DataFrame([{
        "battery_capacity_kwh": payload["battery_capacity_kwh"],
        "start_soc_percent": payload["start_soc_percent"],
        "target_soc_percent": payload["target_soc_percent"],
        "charger_power_kw": payload["charger_power_kw"],
        "ambient_temp_c": payload["ambient_temp_c"],
        "soc_delta_percent": payload["target_soc_percent"] - payload["start_soc_percent"],
        "energy_needed_kwh": energy,
        "vehicle_type": payload["vehicle_type"],
    }])
    pipeline = _bundle["pipeline"]
    minutes = max(float(pipeline.predict(row)[0]), 1.0)

    time_to_80 = None
    if payload["target_soc_percent"] > 80 and payload["start_soc_percent"] < 80:
        row_80 = row.copy()
        row_80["target_soc_percent"] = 80.0
        row_80["soc_delta_percent"] = 80.0 - payload["start_soc_percent"]
        row_80["energy_needed_kwh"] = payload["battery_capacity_kwh"] * (80.0 - payload["start_soc_percent"]) / 100.0
        time_to_80 = max(float(pipeline.predict(row_80)[0]), 1.0)
    return minutes, time_to_80


def _charging_curve(payload, predicted_minutes, time_to_80):
    start = payload["start_soc_percent"]
    target = payload["target_soc_percent"]
    cc_end = min(target, 80.0)
    if start >= 80:
        cc_minutes = 0.0
    elif target <= 80:
        cc_minutes = predicted_minutes
    elif time_to_80 is not None:
        cc_minutes = min(predicted_minutes, time_to_80)
    else:
        cc_minutes = predicted_minutes * ((cc_end - start) / (target - start))
    cv_minutes = max(predicted_minutes - cc_minutes, 0.0)
    points = []
    steps = 24
    for index in range(steps + 1):
        soc = start + (target - start) * index / steps
        if soc <= cc_end and start < 80:
            phase = "CC"
            ratio = (soc - start) / max(cc_end - start, 0.001)
            elapsed = cc_minutes * ratio
        else:
            phase = "CV"
            ratio = (soc - cc_end) / max(target - cc_end, 0.001)
            elapsed = cc_minutes + cv_minutes * (1 - (1 - ratio) ** 2)
        points.append({"soc_percent": round(soc, 1), "minutes": round(elapsed, 1), "phase": phase})
    return {
        "points": points,
        "current_soc_percent": start,
        "soc_80_percent": 80.0 if start < 80 <= target else None,
        "target_soc_percent": target,
        "cc_minutes": round(cc_minutes, 1),
        "cv_minutes": round(cv_minutes, 1),
        "completion_minutes": round(predicted_minutes, 1),
    }


def _battery_care(payload):
    stress = 0
    reasons = []
    if payload["charger_power_kw"] >= 30:
        stress += 2
        reasons.append("high charger power")
    elif payload["charger_power_kw"] >= 22:
        stress += 1
        reasons.append("elevated charger power")
    if payload["target_soc_percent"] > 90:
        stress += 1
        reasons.append("high target SOC")
    if payload["ambient_temp_c"] < 10 or payload["ambient_temp_c"] > 38:
        stress += 1
        reasons.append("temperature outside the preferred range")
    if not payload["battery_preconditioned"] and (payload["ambient_temp_c"] < 10 or payload["ambient_temp_c"] > 38):
        reasons.append("preconditioning is off")

    if stress >= 3:
        level, advice = "HIGH", "Use the highest power only when necessary; precondition and stop at 80% when the trip allows."
    elif stress == 2:
        level, advice = "MODERATE", "Avoid unnecessarily high charging power when there is sufficient time before departure."
    else:
        level, advice = "LOW", "Conditions are relatively gentle; regular mid-range SOC sessions remain the battery-friendly choice."
    return {
        "stress_level": level,
        "stress_score": stress,
        "factors": reasons or ["moderate power and target SOC"],
        "advice": advice,
        "scientific_degradation_model": False,
    }


def _run_prediction(data, charger_override=None):
    if _bundle is None:
        raise RuntimeError("Model not trained yet. Run train_model.py.")
    payload = _normalized_payload(data, charger_override)
    predicted_minutes, time_to_80 = _model_minutes(payload)
    plan = build_plan(payload, predicted_minutes)
    plan["deadline_status"] = "SAFE" if plan["confidence_range_minutes"]["high"] <= payload["departure_hours"] * 60 else "AT RISK"
    plan["safety_buffer_minutes"] = round(payload["departure_hours"] * 60 - plan["confidence_range_minutes"]["high"], 1)
    return {
        "predicted_minutes": round(predicted_minutes, 1),
        "predicted_hours_minutes": _fmt_hm(predicted_minutes),
        "energy_needed_kwh": round(payload["energy_needed_kwh"], 2),
        "avg_effective_power_kw": round(payload["energy_needed_kwh"] / (predicted_minutes / 60), 2),
        "time_to_80_percent_minutes": round(time_to_80, 1) if time_to_80 else None,
        "model_used": _bundle["model_name"],
        "suggestions": build_suggestions(payload, predicted_minutes),
        "plan": plan,
        "curve": _charging_curve(payload, predicted_minutes, time_to_80),
        "battery_care": _battery_care(payload),
        "input": payload,
    }


def _mode_options(data):
    options = []
    for power in AVAILABLE_CHARGER_POWERS:
        result = _run_prediction(data, power)
        options.append({
            "charger_power_kw": power,
            "charging_time_minutes": result["predicted_minutes"],
            "charging_time": result["predicted_hours_minutes"],
            "estimated_cost": result["plan"]["estimated_cost"],
            "estimated_carbon_kg": result["plan"]["estimated_carbon_kg"],
            "deadline_status": result["plan"]["deadline_status"],
            "safety_buffer_minutes": result["plan"]["safety_buffer_minutes"],
        })
    return options


def _select_mode(options, mode):
    if mode == "FAST":
        return min(options, key=lambda item: item["charging_time_minutes"])
    safe = [item for item in options if item["deadline_status"] == "SAFE"]
    if mode in {"ECONOMY", "GREEN"}:
        return min(safe or options, key=lambda item: (item["charger_power_kw"], item["charging_time_minutes"]))
    raise ValueError("mode must be FAST, ECONOMY, or GREEN")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _bundle is not None,
        "model_name": _bundle["model_name"] if _bundle else None,
    })


@app.get("/api/meta")
def meta():
    return jsonify({
        "vehicle_types": VEHICLE_TYPES,
        "charger_presets": CHARGER_PRESETS,
        "metrics": _metrics,
    })


@app.post("/api/predict")
def predict():
    if _bundle is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py."}), 503

    data = request.get_json(force=True, silent=True) or {}
    try:
        result = _run_prediction(data)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error) or "Invalid numeric planning field or value."}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

    return jsonify(result)


@app.post("/api/simulate")
def simulate():
    data = request.get_json(force=True, silent=True) or {}
    scenario = data.get("scenario", {})
    if not isinstance(scenario, dict):
        return jsonify({"error": "scenario must be an object."}), 400
    base = {**data, **{key: value for key, value in scenario.items() if value is not None}}
    try:
        current = _run_prediction(data)
        simulated = _run_prediction(base)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error) or "Invalid simulation input."}), 400
    comparison = [
        {"parameter": "Charging time", "current": current["predicted_hours_minutes"], "scenario": simulated["predicted_hours_minutes"]},
        {"parameter": "Energy required", "current": current["energy_needed_kwh"], "scenario": simulated["energy_needed_kwh"], "unit": "kWh"},
        {"parameter": "Estimated cost", "current": current["plan"]["estimated_cost"], "scenario": simulated["plan"]["estimated_cost"]},
        {"parameter": "Estimated CO2", "current": current["plan"]["estimated_carbon_kg"], "scenario": simulated["plan"]["estimated_carbon_kg"], "unit": "kg"},
        {"parameter": "Deadline status", "current": current["plan"]["deadline_status"], "scenario": simulated["plan"]["deadline_status"]},
    ]
    return jsonify({"current": current, "scenario": simulated, "comparison": comparison})


@app.post("/api/charger-recommendation")
def charger_recommendation():
    data = request.get_json(force=True, silent=True) or {}
    try:
        options = _mode_options(data)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error) or "Invalid charger recommendation input."}), 400
    safe = [option for option in options if option["deadline_status"] == "SAFE"]
    recommended = min(safe or options, key=lambda option: option["charger_power_kw"])
    return jsonify({
        "options": options,
        "recommended": recommended,
        "recommendation_basis": "Lowest available charger power whose conservative finish window meets the deadline; fastest option is used when none can meet it.",
    })


@app.post("/api/charging-plan")
def charging_plan():
    data = request.get_json(force=True, silent=True) or {}
    mode = str(data.get("mode", "FAST")).upper()
    try:
        options = _mode_options(data)
        selected = _select_mode(options, mode)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error) or "Invalid charging plan input."}), 400
    return jsonify({"mode": mode, "selected": selected, "options": options})


@app.post("/api/fleet-schedule")
def fleet_schedule():
    """Schedule vehicles by urgency while keeping the site power bounded."""
    data = request.get_json(force=True, silent=True) or {}
    vehicles = data.get("vehicles", [])
    site_limit_kw = float(data.get("site_limit_kw", 100))
    if not isinstance(vehicles, list) or not vehicles or len(vehicles) > 20:
        return jsonify({"error": "Provide between 1 and 20 vehicles."}), 400
    if site_limit_kw <= 0:
        return jsonify({"error": "site_limit_kw must be positive."}), 400

    try:
        normalized = []
        for index, vehicle in enumerate(vehicles):
            name = str(vehicle.get("name", f"EV-{index + 1}"))
            energy = float(vehicle["energy_needed_kwh"])
            deadline = float(vehicle["deadline_hours"])
            requested_power = float(vehicle.get("charger_power_kw", 7.2))
            if energy <= 0 or deadline <= 0 or requested_power <= 0:
                raise ValueError
            normalized.append({
                "name": name,
                "energy_needed_kwh": energy,
                "deadline_hours": deadline,
                "charger_power_kw": requested_power,
            })
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Each vehicle needs positive energy, deadline, and charger power."}), 400

    normalized.sort(key=lambda vehicle: vehicle["deadline_hours"])
    schedule = []
    used_power = 0.0
    for vehicle in normalized:
        assigned_power = min(vehicle["charger_power_kw"], max(site_limit_kw - used_power, 0.0))
        duration_hours = vehicle["energy_needed_kwh"] / assigned_power if assigned_power else float("inf")
        used_power += assigned_power
        schedule.append({
            **vehicle,
            "assigned_power_kw": round(assigned_power, 2),
            "estimated_duration_hours": round(duration_hours, 2) if assigned_power else None,
            "status": "on time" if duration_hours <= vehicle["deadline_hours"] else "at risk",
        })
    return jsonify({
        "site_limit_kw": site_limit_kw,
        "used_power_kw": round(used_power, 2),
        "available_power_kw": round(max(site_limit_kw - used_power, 0.0), 2),
        "utilization_percent": round(used_power / site_limit_kw * 100, 1),
        "strategy": "Earliest deadline first",
        "schedule": schedule,
        "note": "Power is assigned in earliest-deadline order and never exceeds the configured site limit.",
    })


def _fmt_hm(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(round(minutes % 60))
    if h == 0:
        return f"{m} min"
    return f"{h} hr {m} min"


# ---- serve the frontend -------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
