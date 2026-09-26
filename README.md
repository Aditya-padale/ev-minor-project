# VOLTA — EV Charging Time Prediction and Optimization

Full-stack minor project: ML model + Flask API + web dashboard that predicts
charging time and turns that estimate into a battery-aware charging plan.
VOLTA now answers not only "when will I finish?" but also "what is the best
way to charge before my departure?"

Matches the synopsis: battery capacity, current charge %, ambient temperature
→ predicted charging time, shown in an interactive web app with optimization
tips (per your Introduction/Objectives slides).

## Stack
- **ML**: scikit-learn (RandomForest / GradientBoosting / Ridge, auto-picks best), pandas, numpy
- **Backend**: Flask REST API (`/api/predict`, `/api/meta`, `/api/health`, `/api/simulate`, `/api/charger-recommendation`, `/api/charging-plan`, `/api/fleet-schedule`)
- **Frontend**: vanilla HTML/CSS/JS dashboard (no build step needed)

## Project structure
```
ev-charging-predictor/
├── backend/
│   ├── app.py                  # Flask API + serves frontend
│   ├── requirements.txt
│   └── model/
│       ├── generate_dataset.py # builds physics-informed synthetic dataset
│       ├── train_model.py      # trains + compares models, saves best
│       ├── ev_charging_model.pkl  (pre-trained, included)
│       └── metrics.json           (pre-trained metrics, included)
├── data/
│   └── ev_charging_dataset.csv (pre-generated, included, 12k rows)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## Quick start (model already trained — just run the app)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py
```

Open **http://localhost:5000** — the Flask app serves the frontend directly,
so there's nothing else to start.

## What makes VOLTA different

The original ML prediction remains the core signal, but the planning layer
adds useful decisions around it:

- **Uncertainty-aware timing**: every result includes a likely finish window,
  rather than presenting a single over-precise minute value.
- **Deadline strategy**: departure time is compared with the upper estimate;
  VOLTA recommends an off-peak session, monitoring the session, or the fastest
  available charger.
- **Cost and carbon view**: electricity rate and grid carbon intensity produce
  per-session cost and CO2e estimates.
- **Battery longevity advice**: battery health, temperature, preconditioning,
  charger power, and target SOC are used to recommend gentler charging when
  appropriate.
- **Fleet charging desk**: multiple EVs can be scheduled by deadline while a
  site power limit is respected.
- **Live charging curve**: each prediction includes a CC/CV-shaped curve with
  current SOC, the 80% transition, target SOC, and model-estimated completion.
- **What-if simulation**: temperature, charger power, battery health, target
  SOC, and departure time can be recalculated against the current session.
- **Smart charger recommendation**: 7.2, 11, 22, 30, and 60 kW options are
  evaluated using the same ML prediction and conservative finish window.
- **Charging modes**: FAST selects the shortest calculated session; ECONOMY
  and GREEN select the lowest available power that safely meets the deadline,
  falling back to the least-powerful option when no option is safe.
- **Battery care dashboard**: a transparent rule-based stress indicator uses
  power, temperature, target SOC, and preconditioning. It is advice, not a
  scientific degradation prediction.

These features work offline with user-provided assumptions. Real tariffs,
renewable intensity, charger occupancy, solar generation, and charging logs
can be connected later.

## Retraining the model (optional)

If you want to regenerate the dataset or retrain:

```bash
cd backend/model
python3 generate_dataset.py   # writes ../../data/ev_charging_dataset.csv
python3 train_model.py        # writes ev_charging_model.pkl + metrics.json
```

`train_model.py` compares RandomForest, GradientBoosting and Ridge on a
held-out test split + 5-fold CV, and keeps whichever has the lowest MAE
(currently RandomForest, ~11 min MAE, R² ≈ 0.98 on the synthetic data).

## How the dataset was built

There's no public real-time charger dataset bundled with your synopsis, so
`generate_dataset.py` simulates realistic sessions by numerically integrating
a **CC-CV (Constant Current / Constant Voltage) charging curve**:
- 0–80% SOC: charges near full charger power (the CC phase)
- 80–100% SOC: power tapers linearly down to ~15% (the CV phase) — this is
  why "charging to 100%" takes disproportionately longer than "charging to 80%"
- Ambient temperature derates the effective charge power (cold and very hot
  batteries accept current more slowly)
- Gaussian noise (±6%) is layered on top to imitate real-world measurement
  noise (cable losses, grid sag, connector wear)

This is exactly the kind of relationship your literature review sources
(ensemble ML, deep learning, micro-clustering) are trying to learn from real
logs — swap `data/ev_charging_dataset.csv` for a real dataset (e.g. a Kaggle
EV charging dataset) and rerun `train_model.py` with zero code changes, as
long as you keep the same column names.

## API reference

### `POST /api/predict`
```json
{
  "vehicle_type": "Sedan",
  "battery_capacity_kwh": 55,
  "start_soc_percent": 20,
  "target_soc_percent": 90,
  "charger_power_kw": 30,
  "ambient_temp_c": 28,
  "battery_health_percent": 95,
  "departure_hours": 4,
  "electricity_rate_per_kwh": 8,
  "carbon_intensity_g_per_kwh": 650,
  "battery_preconditioned": false
}
```
→
```json
{
  "predicted_minutes": 87.3,
  "predicted_hours_minutes": "1 hr 27 min",
  "energy_needed_kwh": 38.5,
  "avg_effective_power_kw": 26.47,
  "time_to_80_percent_minutes": 77.2,
  "model_used": "RandomForest",
  "suggestions": [ { "title": "...", "detail": "..." }, ... ],
  "plan": {
    "confidence_range_minutes": {
      "low": 76.8,
      "high": 97.8,
      "label": "1 hr 17 min - 1 hr 38 min"
    },
    "estimated_cost": 308.0,
    "estimated_carbon_kg": 25.03,
    "strategy": "Charge during the lowest-cost window",
    "strategy_detail": "...",
    "battery_health_advice": "..."
  }
}
```

The planning fields are optional for backwards compatibility. Defaults are
95% battery health, four hours to departure, a rate of 8 per kWh, and 650 g
CO2e per kWh. The finish window is a practical estimate based on the model
prediction and a conservative 12% live-condition allowance; it is not a
formal statistical confidence interval.

### `GET /api/meta`
Returns vehicle type list, charger presets, and training metrics — used by
the frontend to populate dropdowns.

### `POST /api/fleet-schedule`
Schedules up to 20 vehicles using earliest-deadline-first ordering:

```json
{
  "site_limit_kw": 50,
  "vehicles": [
    { "name": "EV-01", "energy_needed_kwh": 32, "deadline_hours": 3, "charger_power_kw": 11 },
    { "name": "EV-02", "energy_needed_kwh": 18, "deadline_hours": 1.5, "charger_power_kw": 7.2 }
  ]
}
```

The response reports assigned power, estimated duration, and whether each
vehicle is on time or at risk. For production use, replace the static inputs
with live charger occupancy, tariffs, solar generation, and site telemetry.

### `POST /api/simulate`
Accepts the normal prediction payload plus a `scenario` object. The scenario
can override `ambient_temp_c`, `charger_power_kw`, `battery_health_percent`,
`target_soc_percent`, and `departure_hours`. It returns `current`, `scenario`,
and a comparison table for time, energy, cost, CO2, and deadline status.

### `POST /api/charger-recommendation`
Accepts the normal prediction payload and evaluates the available charger
powers (7.2, 11, 22, 30, and 60 kW). The recommendation is the lowest power
whose conservative finish window meets the deadline, or the fastest option if
none can meet it.

### `POST /api/charging-plan`
Accepts the normal prediction payload and `mode` (`FAST`, `ECONOMY`, or
`GREEN`). It returns every calculated charger option and the selected plan.
All options use the existing ML pipeline; cost and carbon use the supplied
energy, tariff, and grid-intensity assumptions.

## Upgrade flow

The dashboard follows this sequence after the user submits a session:

1. The existing scikit-learn pipeline predicts charging time.
2. The planning layer calculates the finish window, deadline status, cost,
  carbon, CC/CV visualization data, and battery-care advice.
3. Charger powers and charging modes are evaluated by calling the same model
  for each candidate, never by hardcoded result values.
4. The what-if simulator repeats the same calculation with scenario inputs.
5. The fleet desk assigns power in earliest-deadline order while tracking the
  cumulative site limit.

The model is trained on physics-informed synthetic data. Its metrics describe
held-out synthetic samples and are not real-world validation. The model and
`metrics.json` are unchanged by these planning features.

## Mapping back to your synopsis slides
- **Objectives → done**: real-time prediction via web app, ML model, input
  form, optimization suggestions, deadline-aware planning, uncertainty range,
  cost/carbon estimates, battery-health guidance, and a fleet scheduler.
- **Methodology phases**: Phase I = `generate_dataset.py`, Phase II =
  `train_model.py`, Phase III = the rule-based `build_suggestions()` in
  `app.py`, Phase IV = `frontend/`, Phase V = deploy `backend/` (e.g. on
  Render/Railway/PythonAnywhere) — `app.py` already serves the frontend, so
  one deploy target is enough.
- **Budget**: hosting-only cost stands — everything used here (Flask,
  scikit-learn, pandas) is open source, $0 dev cost.

## Notes / next steps
- Swap the synthetic CSV for real charging logs when available (same columns).
- `metrics.json` is regenerated every retrain — surface it in the UI "About"
  section if you want to show model accuracy live in your demo.
- Connect real time-of-use rates, grid carbon intensity, charger occupancy,
  solar output, and actual completed sessions for personalized calibration.
- CORS is enabled in `app.py` for local dev; tighten `CORS(app)` to specific
  origins before any public deployment.
# ev-minor-project
