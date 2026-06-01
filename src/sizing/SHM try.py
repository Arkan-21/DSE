
# --- restructured-project import bootstrap ---
from pathlib import Path as _DSE_Path
import sys as _DSE_sys
_DSE_ROOT = next((p for p in _DSE_Path(__file__).resolve().parents if (p / "src").exists() and (p / "data").exists()), None)
if _DSE_ROOT is not None:
    for _DSE_p in [
        _DSE_ROOT / "src",
        _DSE_ROOT / "src" / "common",
        _DSE_ROOT / "src" / "aerodynamics" / "drag",
        _DSE_ROOT / "src" / "propulsion",
        _DSE_ROOT / "src" / "propulsion" / "engine",
        _DSE_ROOT / "src" / "thermal",
        _DSE_ROOT / "src" / "sizing",
        _DSE_ROOT / "src" / "tanks",
        _DSE_ROOT / "src" / "environment",
        _DSE_ROOT / "src" / "trade_offs",
        _DSE_ROOT / "external",
        _DSE_ROOT / "external" / "pycycle_examples",
    ]:
        if _DSE_p.exists() and str(_DSE_p) not in _DSE_sys.path:
            _DSE_sys.path.insert(0, str(_DSE_p))
# --- end bootstrap ---
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure pandas 
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 100)


class Mach5SafetyFloorSimulator:

    def __init__(self, num_missions=2500, seed=42):  
        self.num_missions = num_missions
        np.random.seed(seed)

        # 1. Physical asset baseline characteristics (Mean wear per standard flight, Std Dev)
        self.base_wear = {
            "TPS": (4.1, 1.1),
            "Structure": (2.8, 0.6),
            "Propulsion": (5.2, 1.5),
        }

        # 2. Operational Environment Matrix (Probabilities and stress scaling multipliers)
        self.profiles = {
            "Low": {"prob": 0.21, "stress": 0.8},
            "Standard": {"prob": 0.682, "stress": 1.1},
            "High": {"prob": 0.108, "stress": 1.5},
        }

        # 3. Geometric Boundary Conditions
        self.hi_safety_floor = 70.0
        total_allowed_decay = 100.0 - self.hi_safety_floor  # 30% structural debt budget

        e_stress = sum(p["prob"] * p["stress"] for p in self.profiles.values())
        stress_variance = sum(p["prob"] * ((p["stress"] - e_stress) ** 2) for p in self.profiles.values())
        stress_std_dev = math.sqrt(stress_variance)

        derived_maint_divisor = (self.base_wear["Propulsion"][0] / self.base_wear["Structure"][0]) + stress_std_dev
        target_fraction_per_flight = (total_allowed_decay / 100.0) / derived_maint_divisor

        self.max_life_budget = {
            sub: (values[0] * e_stress) / target_fraction_per_flight
            for sub, values in self.base_wear.items()
        }

        # Tuned Parameters 
        raw_distribution_drift = ((stress_std_dev * self.base_wear["Structure"][0]) / math.sqrt(self.base_wear["TPS"][0] * self.base_wear["Propulsion"][0])) 
        base_aging_rate = raw_distribution_drift * (total_allowed_decay / 100.0)
        
        self.aging_rate = {
            "TPS": base_aging_rate * 0.8,       
            "Structure": base_aging_rate * 0.75,  
            "Propulsion": base_aging_rate * 0.85, 
        }
        
        self.calculated_floor_intercept = round(total_allowed_decay / self.aging_rate["Propulsion"])

    def calculate_hi_composite(self, current_wear, sub, wear_history, ceiling):
        max_budget = self.max_life_budget[sub]
        physical_hi = max(0.0, ceiling - (current_wear / max_budget * 100.0))

        if len(wear_history) < 3:
            return min(ceiling, max(0.0, physical_hi))

        diffs = np.diff(np.array(wear_history))
        missions = np.arange(len(wear_history))

        monotonicity = np.sum(diffs >= 0) / len(diffs)
        corr = np.corrcoef(missions, np.array(wear_history))[0, 1]
        trendability = max(0.0, corr) if not np.isnan(corr) else 1.0
        residual_variance = np.std(diffs)
        predictability = max(0.1, 1.0 - (residual_variance / 5.0))

        quality_modifier = (monotonicity + trendability + predictability) / 3.0
        composite_hi = physical_hi * (0.8 + 0.2 * quality_modifier)

        return min(ceiling, max(0.0, composite_hi))

    def run_simulation(self, apply_maintenance=True):
        current_wear = {"TPS": 0.0, "Structure": 0.0, "Propulsion": 0.0}
        wear_history = {"TPS": [], "Structure": [], "Propulsion": []}
        health_ceiling = {"TPS": 100.0, "Structure": 100.0, "Propulsion": 100.0}

        maint_intervals = []
        last_maint = 0

        log = []
        decommissioned = False

        for m in range(1, self.num_missions + 1):
            if decommissioned:
                break

            roll = np.random.rand()
            if roll < self.profiles["Low"]["prob"]:
                stress = self.profiles["Low"]["stress"]
                profile = "Low Stress"
            elif roll > 1.0 - self.profiles["High"]["prob"]:
                stress = self.profiles["High"]["stress"]
                profile = "HIGH EXCUR"
            else:
                stress = self.profiles["Standard"]["stress"]
                profile = "Standard"

            for sub, (mean, std) in self.base_wear.items():
                inc = max(0.2, np.random.normal(mean, std) * stress)
                current_wear[sub] += inc
                wear_history[sub].append(current_wear[sub])
                health_ceiling[sub] = max(0.0, health_ceiling[sub] - self.aging_rate[sub])

            hi = {}
            for sub in self.base_wear:
                hi[sub] = self.calculate_hi_composite(current_wear[sub], sub, wear_history[sub], health_ceiling[sub])

            actions = []
            
            if apply_maintenance:
                decom_triggers = [sub for sub in self.base_wear if health_ceiling[sub] <= self.hi_safety_floor]
                if len(decom_triggers) > 0:
                    decommissioned = True
                    action_text = "DECOMMISSIONED"
                else:
                    triggered = False
                    for sub in self.base_wear:
                        if hi[sub] < (health_ceiling[sub] - 15.0):
                            actions.append(sub)
                            triggered = True
                            current_wear[sub] = 0.0
                            wear_history[sub] = []
                            hi[sub] = health_ceiling[sub]
                    
                    if triggered:
                        maint_intervals.append(m - last_maint)
                        last_maint = m
                        action_text = f"Maint ({', '.join(actions)})"
                    else:
                        action_text = "Turnaround Pass"
            else:
                action_text = "No Maint Run"

            log.append({
                "Flight": m, "Profile": profile,
                "TPS_HI": round(hi["TPS"], 1), "Struct_HI": round(hi["Structure"], 1), "Prop_HI": round(hi["Propulsion"], 1),
                "TPS_Ceiling": health_ceiling["TPS"], "Struct_Ceiling": health_ceiling["Structure"], "Prop_Ceiling": health_ceiling["Propulsion"],
                "Action_Taken": action_text,
            })

        return pd.DataFrame(log), maint_intervals


# --- INITIALIZE AND RUN SIMULATIONS ---
sim_no_maint = Mach5SafetyFloorSimulator(num_missions=7, seed=42)
df_no_maint, _ = sim_no_maint.run_simulation(apply_maintenance=False)

sim_lifecycle = Mach5SafetyFloorSimulator(num_missions=2500, seed=42)
df_lifecycle, intervals = sim_lifecycle.run_simulation(apply_maintenance=True)


# --- PRINT METRICS AND DATA TABLE ---
print("=" * 95)
print("             MACH 5 PROTOTYPE LOG MATRIX (ACTIVE MAINTENANCE RUN)")
print("=" * 95)
print(f" -> OPERATIONAL CADENCE: Overhauls autonomously trigger every {np.mean(intervals):.2f} flights.")
print(f" -> CRITICAL INTERCEPT : Propulsion unrecoverable ceiling breaches 70% floor at flight {len(df_lifecycle)}.")
print("-" * 95)
print("PHASE 1: EARLY OPERATIONS & CADENCE TESTING")
print("-" * 95)
print(df_lifecycle.head(15)[["Flight", "Profile", "TPS_HI", "Struct_HI", "Prop_HI", "Action_Taken"]].to_string(index=False))
print("\n" + "-" * 95)
print("PHASE 2: DETAILED MAINTENANCE OVERHAUL EVENTS (SAMPLE ROWS)")
print("-" * 95)
df_maint_only = df_lifecycle[df_lifecycle["Action_Taken"].str.contains("Maint", na=False)]
print(df_maint_only.head(10)[["Flight", "Profile", "TPS_HI", "Struct_HI", "Prop_HI", "Action_Taken"]].to_string(index=False))
print("\n" + "-" * 95)
print("PHASE 3: TERMINAL DECOMMISSIONING INTERCEPT")
print("-" * 95)
print(df_lifecycle.tail(8)[["Flight", "Profile", "TPS_HI", "Struct_HI", "Prop_HI", "Action_Taken"]].to_string(index=False))
print("=" * 95)


# --- PLOT 1: CONTINUOUS DEGRADATION ---
plt.figure(figsize=(11, 4.5))
plt.plot(df_no_maint["Flight"], df_no_maint["TPS_HI"], label="TPS Health Index", color="orangered", marker="o", lw=2)
plt.plot(df_no_maint["Flight"], df_no_maint["Struct_HI"], label="Airframe Structural Index", color="royalblue", marker="s", lw=2)
plt.plot(df_no_maint["Flight"], df_no_maint["Prop_HI"], label="Propulsion Health Index", color="forestgreen", marker="^", lw=2)
plt.axhline(y=70, color="red", linestyle="--", linewidth=1.5, label="Safety Floor Threshold (70%)")
plt.xlabel("Flight Number")
plt.ylabel("Health Indicator (%)")
plt.ylim(-5, 105)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right")
plt.tight_layout()


# --- PLOT 2: MACRO LIFECYCLE WITH EVERY FLOOR CROSSING MARKED ---
plt.figure(figsize=(12, 6))

# 1. Plot the active high-variance zig-zag tracks
#plt.plot(df_lifecycle["Flight"], df_lifecycle["TPS_HI"], label="TPS Active Track", color="orangered", alpha=0.25, lw=0.5)
plt.plot(df_lifecycle["Flight"], df_lifecycle["Prop_HI"], label="Propulsion Active Track", color="forestgreen", alpha=0.25, lw=0.5)
plt.plot(df_lifecycle["Flight"], df_lifecycle["Struct_HI"], label="Airframe Active Track", color="royalblue", alpha=0.25, lw=0.5)

# 2. Plot the long-term unrecoverable ceilings
#plt.plot(df_lifecycle["Flight"], df_lifecycle["TPS_Ceiling"], label="TPS Structural Boundary", color="orangered", linestyle="-", lw=2)
plt.plot(df_lifecycle["Flight"], df_lifecycle["Struct_Ceiling"], label="Airframe Structural Boundary", color="royalblue", linestyle="-", lw=2)
plt.plot(df_lifecycle["Flight"], df_lifecycle["Prop_Ceiling"], label="Propulsion Structural Boundary", color="forestgreen", linestyle="-", lw=2.5)

# 3. Reference Airworthiness Limit Line
plt.axhline(y=70, color="red", linestyle="--", linewidth=1.8, label="Airworthiness Safety Limit (70% Floor)")

# DYNAMIC CROSSING IDENTIFICATION & MARKERS
# Create a boolean filter to isolate every instance where an active track dips to or below 70%
crossings_mask = (df_lifecycle["TPS_HI"] <= 70.0) | (df_lifecycle["Struct_HI"] <= 70.0) | (df_lifecycle["Prop_HI"] <= 70.0)
df_crossings = df_lifecycle[crossings_mask]

# Plot a distinct marker point for every single identified crossing event
plt.scatter(
    df_crossings["Flight"], 
    [70.0] * len(df_crossings), 
    color="purple", 
    edgecolor="black", 
    s=35, 
    zorder=6, 
    label=f"Floor Crossings (First event: 534)"
)

# 4. Terminal ceiling intercept indicators
intercept_x = len(df_lifecycle)
plt.axvline(x=intercept_x, color="black", linestyle=":", linewidth=1.5, alpha=0.7)
plt.axhline(y=70.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7)

plt.scatter(intercept_x, 70.0, color="black", edgecolor="red", s=45, zorder=5)
plt.annotate(f" Intercept: Flight {intercept_x}", (intercept_x, 70.5), fontsize=9, fontweight="bold", color="black")

# 5. Presentation Parameters
plt.xlabel("Number of Flight Missions")
plt.ylabel("Available Structural Capacity (% of initial health)")
plt.xlim(0, len(df_lifecycle) + 20)
plt.ylim(65, 103)
plt.grid(True, linestyle=":", alpha=0.5)

plt.legend(loc="upper right", ncol=2, fontsize=9)
plt.tight_layout()

plt.show() 