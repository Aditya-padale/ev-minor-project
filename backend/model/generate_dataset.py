"""
generate_dataset.py
--------------------
Builds a synthetic-but-physics-informed EV charging dataset.

Real charging follows a CC-CV (Constant Current / Constant Voltage) curve:
- 0% -> ~80% SOC charges near-linearly at (roughly) full charger power,
  derated by ambient temperature.
- 80% -> 100% SOC tapers off (CV phase) because the battery management
  system throttles current to protect the cells, so it takes
  disproportionately longer per % gained.

We simulate many (battery, charger, temperature, vehicle) combinations,
integrate the CC-CV curve numerically to get a ground-truth charging time,
then add realistic sensor/behavioural noise so an ML model has something
non-trivial to learn (this mirrors what a real logged-session dataset
would look like).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 12000

VEHICLE_TYPES = ["Hatchback", "Sedan", "SUV", "Two-Wheeler"]

# (min_capacity_kWh, max_capacity_kWh, charge_efficiency)
VEHICLE_PROFILES = {
    "Two-Wheeler": (2.0, 4.5, 0.92),
    "Hatchback": (25.0, 40.0, 0.90),
    "Sedan": (40.0, 65.0, 0.90),
    "SUV": (65.0, 100.0, 0.88),
}

CHARGER_TYPES = {
    "Slow AC (3.3 kW)": 3.3,
    "Standard AC (7.2 kW)": 7.2,
    "Fast AC (11 kW)": 11.0,
    "DC Fast (30 kW)": 30.0,
    "DC Ultra-Fast (60 kW)": 60.0,
}


def temperature_derate(temp_c: float) -> float:
    """Battery charge acceptance drops in cold weather and (mildly) in
    extreme heat due to BMS thermal throttling. Peak efficiency ~20-30C."""
    if temp_c < 0:
        return 0.55
    if temp_c < 10:
        return 0.55 + (temp_c / 10.0) * 0.25          # 0.55 -> 0.80
    if temp_c < 20:
        return 0.80 + ((temp_c - 10) / 10.0) * 0.15    # 0.80 -> 0.95
    if temp_c <= 35:
        return 1.0
    if temp_c <= 45:
        return 1.0 - ((temp_c - 35) / 10.0) * 0.20     # 1.0 -> 0.80
    return 0.65


def simulate_charge_time_minutes(capacity_kwh, start_soc, target_soc,
                                  charger_kw, temp_c, efficiency):
    """Numerically integrate a CC-CV charging curve in 0.5% SOC steps."""
    derate = temperature_derate(temp_c)
    effective_power = charger_kw * derate * efficiency

    soc = start_soc
    minutes = 0.0
    step = 0.5  # percent per integration step
    while soc < target_soc:
        this_step = min(step, target_soc - soc)
        if soc < 80:
            power_factor = 1.0                     # CC phase: full power
        else:
            # CV taper: power ramps down linearly from 100% at 80% SOC
            # to ~15% at 100% SOC
            power_factor = 1.0 - 0.85 * ((soc - 80) / 20.0)
            power_factor = max(power_factor, 0.15)

        kw_now = max(effective_power * power_factor, 0.05)
        energy_needed = capacity_kwh * (this_step / 100.0)
        minutes += (energy_needed / kw_now) * 60.0
        soc += this_step
    return minutes


def build_dataset(n=N_SAMPLES) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        vtype = RNG.choice(VEHICLE_TYPES)
        cap_min, cap_max, eff = VEHICLE_PROFILES[vtype]
        capacity = RNG.uniform(cap_min, cap_max)

        charger_name = RNG.choice(list(CHARGER_TYPES.keys()))
        charger_kw = CHARGER_TYPES[charger_name]

        start_soc = RNG.uniform(2, 70)
        target_soc = RNG.uniform(start_soc + 5, 100)
        temp_c = RNG.uniform(-5, 48)

        true_minutes = simulate_charge_time_minutes(
            capacity, start_soc, target_soc, charger_kw, temp_c, eff
        )

        # Behavioural / measurement noise: cable losses, connector wear,
        # grid sag at peak hours, user-reported temp rounding, etc.
        noisy_minutes = true_minutes * RNG.normal(1.0, 0.06)
        noisy_minutes = max(noisy_minutes, 1.0)

        rows.append({
            "vehicle_type": vtype,
            "battery_capacity_kwh": round(capacity, 2),
            "start_soc_percent": round(start_soc, 1),
            "target_soc_percent": round(target_soc, 1),
            "charger_power_kw": charger_kw,
            "ambient_temp_c": round(temp_c, 1),
            "charging_time_minutes": round(noisy_minutes, 1),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    out_path = "/home/claude/ev-charging-predictor/data/ev_charging_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")
    print(df.select_dtypes("number").describe())
