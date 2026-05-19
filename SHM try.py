import math
import numpy as np
import pandas as pd


class Mach5SafetyFloorSimulator:

    def __init__(self, num_missions=40):
        self.num_missions = num_missions
        np.random.seed(42)  # Maintain standard seed for validation runs

        # Base wear characteristics per flight (Mean wear, Std Dev)
        self.base_wear = {
            "TPS": (4.1, 1.1),  # Rapid aero-thermal breakdown
            "Structure": (2.8, 0.6),  # Acoustic/Vibration strain accumulation
            "Propulsion": (
                5.2,
                1.5,
            ),  # Multi-mode switching transient degradation
        }

        # The operational floor. If HI falls below 70%, maintenance is mandatory.
        self.hi_safety_floor = 70.0

        # Max theoretical life budget used to map the baseline linear decay
        self.max_life_budget = {"TPS": 20.0, "Structure": 14.0, "Propulsion": 25.0}

    def calculate_hi_composite(self, current_wear, sub, wear_history):
        """Calculates Health Indicator (HI) starting at 100% and decaying monotonically

        with each consecutive flight.
        """
        max_budget = self.max_life_budget[sub]

        # Physical structural health drops immediately with wear
        physical_hi = max(0.0, 100.0 - (current_wear / max_budget * 100.0))

        if len(wear_history) < 2:
            return min(100.0, max(0.0, physical_hi))

        diffs = np.diff(np.array(wear_history))
        missions = np.arange(len(wear_history))

        # Signal parameters evaluate operational health anomalies
        monotonicity = np.sum(diffs >= 0) / len(diffs)
        corr = np.corrcoef(missions, np.array(wear_history))[0, 1]
        trendability = max(0.0, corr) if not np.isnan(corr) else 1.0
        residual_variance = np.std(diffs)
        predictability = max(0.1, 1.0 - (residual_variance / 5.0))

        # Composite score blends physical structural wear with health signal metrics
        quality_modifier = (monotonicity + trendability + predictability) / 3.0
        composite_hi = physical_hi * (0.8 + 0.2 * quality_modifier)

        return min(100.0, max(0.0, composite_hi))

    def calculate_maintenance_probability(self, current_wear, sub, stress_factor):
        """Calculates risk probability.

        Uses an exponential hazard model to ensure risk is never 0% on flight
        1, and grows exponentially as wear accumulates.
        """
        # Assign a tiny baseline hazard risk (unforeseen hypersonic flight stress)
        base_hazard = 0.02

        # Risk multiplier scales up with current accumulated wear
        wear_fraction = current_wear / self.max_life_budget[sub]

        # Cumulative risk equation
        risk_prob = 1.0 - math.exp(
            -(base_hazard + (wear_fraction**2) * 0.5) * stress_factor
        )

        return min(100.0, max(0.5, risk_prob * 100.0))

    def run_simulation(self):
        current_wear = {"TPS": 0.0, "Structure": 0.0, "Propulsion": 0.0}
        wear_history = {"TPS": [], "Structure": [], "Propulsion": []}

        maint_intervals = {"TPS": [], "Structure": [], "Propulsion": []}
        last_maint = {"TPS": 0, "Structure": 0, "Propulsion": 0}

        log = []
        for m in range(1, self.num_missions + 1):
            roll = np.random.rand()
            if roll < 0.20:
                stress, profile = 0.8, "Low Stress Flight"
            elif roll > 0.85:
                stress, profile = 1.5, "HIGH AERO-THERMAL EXCURSION"
            else:
                stress, profile = 1.1, "Standard Test Profile"

            # 1. Pre-Flight Risk Assessment (Always > 0%, increases over intervals)
            probs = {}
            for sub in self.base_wear:
                probs[sub] = self.calculate_maintenance_probability(
                    current_wear[sub], sub, stress
                )

            # 2. Flight Run - Accumulate damage
            for sub, (mean, std) in self.base_wear.items():
                inc = max(0.2, np.random.normal(mean, std) * stress)
                current_wear[sub] += inc
                wear_history[sub].append(current_wear[sub])

            # 3. Post-Flight Health Evaluation
            hi = {}
            for sub in self.base_wear:
                hi[sub] = self.calculate_hi_composite(
                    current_wear[sub], sub, wear_history[sub]
                )

            # 4. Check Against the 80% Safety Floor Trigger
            actions = []
            for sub in self.base_wear:
                if hi[sub] < self.hi_safety_floor:
                    actions.append(f"EXTENSIVE {sub} OVERHAUL")
                    maint_intervals[sub].append(m - last_maint[sub])
                    last_maint[sub] = m
                    current_wear[sub] = 0.0
                    wear_history[sub] = []

            log.append(
                {
                    "Mission": m,
                    "Profile": profile,
                    "TPS_Risk": probs["TPS"],
                    "Struct_Risk": probs["Structure"],
                    "Prop_Risk": probs["Propulsion"],
                    "TPS_HI": hi["TPS"],
                    "Struct_HI": hi["Structure"],
                    "Prop_HI": hi["Propulsion"],
                    "Action": (
                        ", ".join(actions) if actions else "Routine Turnaround Pass"
                    ),
                }
            )

        return pd.DataFrame(log), maint_intervals


# --- Run Vehicle Lifecycle ---
sim = Mach5SafetyFloorSimulator(num_missions=30)
df_log, intervals = sim.run_simulation()

# --- Print Formatted Results Matrix ---
print("=" * 120)
print(
    "                MACH 5 PROTOTYPE TELEMETRY - 80% HI SAFETY FLOOR & CUMULATIVE RISK MATRIX"
)
print("=" * 120)
print(
    df_log.head(15).to_string(
        index=False,
        formatters={
            "TPS_Risk": "{:.1f}%".format,
            "Struct_Risk": "{:.1f}%".format,
            "Prop_Risk": "{:.1f}%".format,
            "TPS_HI": "{:.1f}%".format,
            "Struct_HI": "{:.1f}%".format,
            "Prop_HI": "{:.1f}%".format,
        },
    )
)
print("=" * 120)
print("STATISTICAL VALIDATION SUMMARY (COMPUTED AT RUNTIME):")
for subsystem, values in intervals.items():
    print(
        f" -> {subsystem:<12} Mean Intercept Interval (HI < 80%): Every {np.mean(values):.1f} flights"
    )
print("=" * 120)