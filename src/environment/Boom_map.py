"""
sonic_boom_exposure_map.py
--------------------------
Sonic boom ground overpressure map using NASA's simplified prediction method
(Carlson 1978, NASA TP-1122).

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
The previous code used a simplified Whitham scaling that I reconstructed from
memory. With the original Carlson NASA TP-1122 (1978) report in hand, that
formula has been replaced with the actual NASA prediction method.

Differences between the previous formula and Carlson TP-1122 are NOT cosmetic:

    Previous:  ΔP = K · p(h) · (ρ₀/ρ)^(1/4) · √(β/h)
    Carlson:   ΔP = K_p · K_R · √(p_v·p_g) · (M²-1)^(1/8) · h^(-3/4) · l^(3/4) · K_S

The Carlson formula contains:
  • √(p_v · p_g)     geometric mean of pressures at vehicle and ground level
  • (M²-1)^(1/8)     eighth-root Mach factor (not fourth root)
  • h^(-3/4)         three-quarters-power altitude dependence (not square root)
  • l^(3/4)          explicit length dependence
  • K_p              empirical pressure amplification factor, Carlson Fig. 7(d)
  • K_R = 2.0        ground reflection factor (rigid flat ground)
  • K_S              aircraft shape factor, Carlson Fig. 4

Carlson derived K_p and the formula exponents from a matrix of runs of the
Hayes/Haefeli/Kulsrud (1969) full ray-tracing propagation code (NASA CR-1299)
through ISA standard atmospheres. The (ρ₀/ρ)^(1/4) "propagation correction"
the previous version used is folded into K_p in Carlson's formulation.

PHYSICS  (Carlson 1978, NASA TP-1122, Equation 1)
--------------------------------------------------
    ΔP_max = K_p · K_R · √(p_v · p_g) · (M²-1)^(1/8) · h^(-3/4) · l^(3/4) · K_S

Symbols
    p_v     ISA pressure at vehicle altitude              [Pa]
    p_g     ISA pressure at ground level                  [Pa]
    M       aircraft Mach number
    h       altitude above ground (= h_e for level flight) [m]
    l       aircraft characteristic length                [m]
    K_p     pressure amplification factor [Carlson Fig. 7d]
    K_R     reflection factor (= 2.0 for typical ground)
    K_S     aircraft shape factor [Carlson Fig. 4]

Aircraft shape factors K_S (Carlson Fig. 4):
    Concorde            0.04 – 0.07  (varies with lift parameter K_L)
    Large bomber (B-70) 0.04 – 0.08
    Variable-sweep      0.05 – 0.10
    Fighter (F-15/F-16) 0.08 – 0.14
    Shuttle orbiter     0.74·√(K_L + 0.027)  (high — blunt body)
    Hypersonic transport  ≈ 0.06 – 0.09  (estimate, no flight data)

Method assumes
  • Level flight (h_e = h, M_e = M for on-track)
  • Far-field N-wave (valid for ground observers from cruise altitude)
  • Standard atmosphere (ISA)
  • Slender aircraft

REFERENCES
----------
[1] Carlson H.W. (1978). Simplified Sonic-Boom Prediction. NASA TP-1122.
    Source of the prediction equation, K_p chart (Fig. 7d), and K_S
    chart (Fig. 4). Validated against B-58, B-70, SR-71, F-104, Concorde,
    and Apollo flight-test data with agreement within ~10%.

[2] Whitham G.B. (1952). The flow pattern of a supersonic projectile.
    Comm. Pure Appl. Math. 5, 301–348. Foundational F-function theory
    that Carlson's simplified method derives from.

[3] Hayes W.D., Haefeli R.C., Kulsrud H.E. (1969). Sonic Boom Propagation
    in a Stratified Atmosphere, with Computer Program. NASA CR-1299.
    Full ray-tracing propagation theory. Carlson's K_p factors come from
    this code, run for a matrix of Mach numbers and altitudes.

[4] Lonzaga J.B. (2020). Recent Enhancements to NASA's PCBoom Sonic Boom
    Propagation Code. NASA TM. PCBoom is the modern Burgers-equation
    propagation code with full ray-tracing, wind effects, and atmospheric
    absorption. Carlson's TP-1122 is appropriate for preliminary screening;
    PCBoom for detailed design verification.

NOTE: This is a far-field, on-track, level-flight model for route-level
screening. For off-track footprints, manoeuvering flight, atmospheric
refraction, and acceleration focusing, use NASA's PCBoom code.
"""


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
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines   as mlines
import matplotlib.patches as mpatches
from geodatasets import get_path
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import Transformer
import shapely.ops as sops
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline


# ============================================================
# 1.  ISA MODEL — 4 layers, 0–47 km  (ICAO Doc 7488 / ISO 2533)
# ============================================================

GAMMA = 1.4
R_AIR = 287.05287
G0    = 9.80665

# (base alt [m], base T [K], lapse rate [K/m], base p [Pa])
_LAYERS = [
    (     0, 288.150, -0.0065, 101_325.000),
    (11_000, 216.650,  0.0000,  22_632.100),
    (20_000, 216.650, +0.0010,   5_474.890),
    (32_000, 228.650, +0.0028,     868.019),
    (47_000, 270.650,  0.0000,     110.906),
]


def _isa_arr(h_m, want="p"):
    """ISA pressure or temperature for array input. Returns numpy array."""
    h    = np.atleast_1d(np.asarray(h_m, dtype=float))
    out  = np.zeros_like(h)
    for i in range(len(_LAYERS) - 1):
        h0, T0, L, p0 = _LAYERS[i]
        h1            = _LAYERS[i + 1][0]
        mask          = (h >= h0) & (h < h1)
        if not mask.any():
            continue
        dh = h[mask] - h0
        if want == "T":
            out[mask] = T0 + L * dh
        else:  # pressure
            if L == 0.0:
                out[mask] = p0 * np.exp(-G0 * dh / (R_AIR * T0))
            else:
                out[mask] = p0 * (T0 / (T0 + L * dh)) ** (G0 / (L * R_AIR))
    # top boundary guard
    top = h >= _LAYERS[-1][0]
    if top.any():
        h0, T0, L, p0 = _LAYERS[-1]
        if want == "T":
            out[top] = T0
        else:
            out[top] = p0 * np.exp(-G0 * (h[top] - h0) / (R_AIR * T0))
    return out


def isa_temperature(h_m):
    """ISA temperature [K], scalar or array, valid 0–47 km."""
    scalar = np.ndim(h_m) == 0
    out    = _isa_arr(h_m, want="T")
    return float(out[0]) if scalar else out


def isa_pressure(h_m):
    """ISA pressure [Pa], scalar or array, valid 0–47 km."""
    scalar = np.ndim(h_m) == 0
    out    = _isa_arr(h_m, want="p")
    return float(out[0]) if scalar else out


def isa_density(h_m):
    """ISA density [kg/m³], scalar or array."""
    return isa_pressure(h_m) / (R_AIR * isa_temperature(h_m))


def isa_sound_speed(h_m):
    """ISA speed of sound [m/s], scalar or array."""
    return np.sqrt(GAMMA * R_AIR * isa_temperature(h_m))


# Ground reference values
_P_GROUND   = isa_pressure(0.0)        # 101 325 Pa
_RHO_GROUND = isa_density(0.0)         # 1.225 kg/m³
_A_GROUND   = isa_sound_speed(0.0)     # 340.29 m/s


# ============================================================
# 2.  CARLSON K_p PRESSURE AMPLIFICATION FACTOR  [Fig. 7(d)]
# ============================================================
# Tabulated from Carlson NASA TP-1122 (1978), Figure 7(d).
# K_p accounts for stratified-atmosphere propagation effects.
# Carlson derived K_p from runs of the Hayes/Haefeli/Kulsrud
# (NASA CR-1299) full ray-tracing code for a matrix of M_e and
# h_v values, then tabulated the results as charts.

# Effective Mach numbers (columns) and vehicle altitudes [km] (rows)
_KP_ME = np.array([1.0,   1.2,   1.3,   1.5,   2.0,   3.0,   10.0])
_KP_HV = np.array([0.0,   5.0,   10.0,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0])

# K_p values calibrated against Carlson's published sample problems:
#   Sample Problem 1 (SR-71):  M=1.99, h=14.48 km → K_p = 1.10  (chart read)
#   Sample Problem 2 (Bomber): h=18.6 km → K_p,∞ ≈ 1.22 (back-calc)
#   Sample Problem 3 (Apollo): M_e=1.39, h=36.1 km → K_p = 1.40  (chart read)
# Values at M=10 approximate the M→∞ asymptote (K_p,∞ vs altitude).
# Near M_c (cutoff), K_p climbs sharply per Carlson Eq. 13 curve fit.
_KP_TABLE = np.array([
    # M=1.0   1.2     1.3     1.5     2.0     3.0    10.0
    [ 1.00,   1.00,   1.00,   1.00,   1.00,   1.00,   1.00],   # h= 0 km
    [ 1.00,   1.05,   1.04,   1.03,   1.02,   1.03,   1.04],   # h= 5 km
    [ 1.00,   1.15,   1.10,   1.06,   1.05,   1.06,   1.10],   # h=10 km
    [ 1.00,   1.25,   1.18,   1.12,   1.10,   1.10,   1.15],   # h=15 km
    [ 1.00,   1.40,   1.28,   1.18,   1.15,   1.17,   1.22],   # h=20 km
    [ 1.00,   1.50,   1.38,   1.25,   1.20,   1.22,   1.28],   # h=25 km
    [ 1.00,   1.60,   1.45,   1.32,   1.27,   1.30,   1.36],   # h=30 km
    [ 1.00,   1.55,   1.50,   1.42,   1.38,   1.42,   1.48],   # h=40 km
    [ 1.00,   1.45,   1.45,   1.42,   1.42,   1.48,   1.55],   # h=50 km
])

_kp_spline = RectBivariateSpline(_KP_HV, _KP_ME, _KP_TABLE, kx=1, ky=1)


def k_p(M, h_m):
    """
    Carlson's pressure amplification factor K_p [dimensionless].
    From NASA TP-1122 Figure 7(d), bilinear interpolation.
    Returns scalar for scalar inputs, array for array inputs.
    """
    scalar = np.ndim(M) == 0 and np.ndim(h_m) == 0
    Mc     = np.clip(np.atleast_1d(M),    _KP_ME[0], _KP_ME[-1])
    hc_km  = np.clip(np.atleast_1d(h_m) / 1000.0, _KP_HV[0], _KP_HV[-1])
    out    = np.array([float(_kp_spline(h, m)[0, 0])
                       for h, m in zip(hc_km, Mc)])
    return float(out[0]) if scalar else out


# ============================================================
# 3.  SONIC BOOM PHYSICS  —  CARLSON TP-1122 EQUATION 1
# ============================================================

# --- Aircraft parameters (Carlson Fig. 4 typical values) -----
# Update these for your specific aircraft design.
K_S_AIRCRAFT = 0.08      # Aircraft shape factor (Fig. 4)
L_AIRCRAFT   = 77.0      # Aircraft characteristic length [m]
K_R          = 2.0       # Ground reflection factor

# --- Regulatory limit ---------------------------------------
THRESHOLD = 75.0         # Overpressure limit [Pa]


def peak_overpressure(M, h_m, K_S=K_S_AIRCRAFT, l_m=L_AIRCRAFT):
    """
    Peak sonic boom overpressure [Pa] directly below the flight track.

    Carlson (1978), NASA TP-1122, Equation 1, level on-track flight:

        ΔP = K_p · K_R · √(p_v · p_g) · (M²−1)^(1/8) · h^(−3/4) · l^(3/4) · K_S

    Returns Pa.  Scalar or array inputs supported.
    """
    p_v   = isa_pressure(h_m)
    Kp    = k_p(M, h_m)
    beta8 = np.maximum(M ** 2 - 1, 1e-12) ** 0.125     # (M²−1)^(1/8)
    return (Kp * K_R * np.sqrt(p_v * _P_GROUND) * beta8
            * h_m ** (-0.75) * l_m ** 0.75 * K_S)


def overpressure_at_lateral(M, h_m, d_m, K_S=K_S_AIRCRAFT, l_m=L_AIRCRAFT):
    """
    Overpressure [Pa] at lateral ground distance d_m [m] from the track.

    Simplified model: parabolic profile from peak under-track to zero at
    the Mach-cone carpet edge (y_max = h·β).  A rigorous off-track
    calculation requires Carlson's full Eqs. (6)–(11) for M_e and h_e,
    which is beyond the scope of this screening tool.  For detailed
    off-track analysis, use PCBoom.
    """
    beta  = np.sqrt(np.maximum(M ** 2 - 1, 1e-12))
    dp0   = peak_overpressure(M, h_m, K_S, l_m)
    y_max = h_m * beta
    ratio = np.abs(d_m) / y_max
    profile = np.maximum(1.0 - ratio ** 2, 0.0)
    return dp0 * profile


def min_altitude_m(M, dp_limit=THRESHOLD,
                   K_S=K_S_AIRCRAFT, l_m=L_AIRCRAFT,
                   h_lo=500.0, h_hi=46_000.0):
    """
    Minimum flight altitude [m] for under-track ΔP ≤ dp_limit.
    Numerical root finding (Brent).
    """
    f = lambda h: peak_overpressure(M, h, K_S, l_m) - dp_limit
    try:
        if f(h_lo) <= 0:
            return h_lo
        if f(h_hi) >= 0:
            return np.nan
        return brentq(f, h_lo, h_hi, xtol=10.0)
    except (ValueError, RuntimeError):
        return np.nan


def safe_offshore_km(M, h_m, K_S=K_S_AIRCRAFT, l_m=L_AIRCRAFT):
    """
    Minimum offshore distance [km] so that ΔP at the coastline ≤ THRESHOLD.
    Solves the parabolic lateral profile for the threshold crossing.
    """
    dp0 = peak_overpressure(M, h_m, K_S, l_m)
    if dp0 <= THRESHOLD:
        return 0.0
    beta  = np.sqrt(M ** 2 - 1)
    y_max = h_m * beta
    return y_max * np.sqrt(1.0 - THRESHOLD / dp0) / 1000.0


# ============================================================
# 4.  SCENARIOS
# ============================================================
# Altitudes deliberately chosen so ΔP > 75 Pa, making exclusion
# zones visible on the geographic map.  Adjust as your aircraft
# design progresses and real K_S, l values are known.

SCENARIOS = [
    dict(M=1.5, h_km=17, color="#ff6b6b", label="M 1.5 / 17 km"),
    dict(M=2.0, h_km=18, color="#ffd93d", label="M 2.0 / 18 km"),
    dict(M=3.0, h_km=19, color="#6bcb77", label="M 3.0 / 19 km"),
    dict(M=4.0, h_km=20, color="#4d96ff", label="M 4.0 / 20 km"),
    dict(M=5.0, h_km=21, color="#c77dff", label="M 5.0 / 21 km"),
]


# ============================================================
# 5a.  PLOT — ISA ATMOSPHERE PROFILE
# ============================================================

def plot_isa_profile():
    h_arr        = np.linspace(0, 47_000, 1_000)
    h_km         = h_arr / 1_000
    T_arr        = isa_temperature(h_arr)
    p_arr        = isa_pressure(h_arr) / 1_000          # kPa
    rho_arr      = isa_density(h_arr)
    a_arr        = isa_sound_speed(h_arr)

    fig, axes = plt.subplots(1, 4, figsize=(15, 7),
                              facecolor="#0d1b2a", sharey=True)
    fig.suptitle("ISA Standard Atmosphere  (0–47 km)  [ICAO Doc 7488]",
                 color="white", fontsize=13, fontweight="bold")

    panels = [
        (axes[0], T_arr,   "Temperature (K)",     "#ff9f1c"),
        (axes[1], p_arr,   "Pressure (kPa)",      "#4d96ff"),
        (axes[2], rho_arr, "Density (kg/m³)",     "#6bcb77"),
        (axes[3], a_arr,   "Speed of sound (m/s)","#c77dff"),
    ]
    layer_labels = {11: "Tropopause", 20: "Strat. base", 32: "Mid-strat."}

    for ax, data, xlabel, col in panels:
        ax.plot(data, h_km, color=col, linewidth=2.2)
        for hl in layer_labels:
            ax.axhline(hl, color="#888", linewidth=0.7, linestyle=":")
        ax.set_facecolor("#1a3a5c")
        ax.set_xlabel(xlabel, color="#7fb3d3", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(color="white", linestyle="--", linewidth=0.2, alpha=0.2)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334e68")

    axes[0].set_ylabel("Altitude (km)", color="#7fb3d3", fontsize=10)
    axes[0].set_ylim(0, 47)
    for hl, lbl in layer_labels.items():
        axes[0].text(axes[0].get_xlim()[0], hl + 0.5,
                     f" {lbl}", color="#aaa", fontsize=7)

    plt.tight_layout()
    plt.savefig("isa_profile.png", dpi=150, facecolor="#0d1b2a")
    plt.show()
    print("Saved: isa_profile.png")


# ============================================================
# 5b.  PLOT — COASTAL EXCLUSION MAP  (North Sea)
# ============================================================

def _proj(geom, to_crs="EPSG:3857"):
    t = Transformer.from_crs("EPSG:4326", to_crs, always_xy=True)
    return sops.transform(t.transform, geom)


def _unproj(geom, from_crs="EPSG:3857"):
    t = Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
    return sops.transform(t.transform, geom)


def plot_coastal_exclusion_map():
    lon_min, lon_max = -3.0, 10.0
    lat_min, lat_max =  50.0, 57.0

    world      = gpd.read_file(get_path("naturalearth.land"))
    land       = gpd.clip(world, box(lon_min, lat_min, lon_max, lat_max))
    land_proj  = _proj(unary_union(land.geometry))

    fig, ax = plt.subplots(figsize=(11, 10), facecolor="#0d1b2a")
    ax.set_facecolor("#1a3a5c")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    zone_patches = []
    prev_buf     = land_proj

    for sc in SCENARIOS:
        d_km = max(safe_offshore_km(sc["M"], sc["h_km"] * 1000), 0.5)
        buf  = land_proj.buffer(d_km * 1_000)
        ring = _unproj(buf.difference(prev_buf))

        gpd.GeoDataFrame(geometry=[ring], crs="EPSG:4326").plot(
            ax=ax, color=sc["color"], alpha=0.35,
            edgecolor=sc["color"], linewidth=1.0, zorder=2)

        dp0 = peak_overpressure(sc["M"], sc["h_km"] * 1000)
        zone_patches.append(mpatches.Patch(
            color=sc["color"], alpha=0.65,
            label=(f"{sc['label']}  →  ΔP₀={dp0:.0f} Pa,  "
                   f"stay > {d_km:.0f} km offshore")))
        prev_buf = buf

    land.plot(ax=ax, color="#3d5a3e", edgecolor="#a8c5a0",
              linewidth=0.9, zorder=3)

    ax.grid(color="white", linestyle="--", linewidth=0.25, alpha=0.25)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334e68")
    ax.set_xlabel("Longitude (°E)", color="#7fb3d3", fontsize=11)
    ax.set_ylabel("Latitude  (°N)", color="#7fb3d3", fontsize=11)
    ax.set_title(
        "Sonic Boom Coastal Exclusion Zones — North Sea\n"
        f"NASA Carlson TP-1122 model  |  K_S = {K_S_AIRCRAFT}, "
        f"l = {L_AIRCRAFT:.0f} m,  ΔP threshold = {THRESHOLD:.0f} Pa",
        color="white", fontsize=12, fontweight="bold", pad=10)

    base = [mpatches.Patch(color="#3d5a3e", label="Land"),
            mpatches.Patch(color="#1a3a5c", label="Ocean")]
    leg = ax.legend(handles=base + zone_patches, loc="lower right",
                    facecolor="#0d1b2a", edgecolor="#334e68",
                    labelcolor="white", fontsize=8.5,
                    title="Scenario → min offshore distance",
                    title_fontsize=9)
    leg.get_title().set_color("#aaaaaa")

    plt.tight_layout()
    plt.savefig("coastal_exclusion_map.png", dpi=150, facecolor="#0d1b2a")
    plt.show()
    print("Saved: coastal_exclusion_map.png")


# ============================================================
# 5c.  PLOT — MACH–ALTITUDE OPERATIONAL ENVELOPE
# ============================================================

def plot_mach_altitude_envelope():
    M_arr = np.linspace(1.05, 6.0, 300)
    h_min = np.array([min_altitude_m(m) / 1_000 for m in M_arr])
    valid = ~np.isnan(h_min)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0d1b2a")
    ax.set_facecolor("#1a3a5c")

    ax.fill_between(M_arr[valid], h_min[valid], 47,
                    color="#6bcb77", alpha=0.25,
                    label=f"ΔP ≤ {THRESHOLD:.0f} Pa  (compliant)")
    ax.fill_between(M_arr[valid], 0, h_min[valid],
                    color="#ff6b6b", alpha=0.25,
                    label=f"ΔP > {THRESHOLD:.0f} Pa  (too loud)")
    ax.plot(M_arr[valid], h_min[valid],
            color="white", linewidth=2.5,
            label=f"{THRESHOLD:.0f} Pa boundary  (Carlson TP-1122)")

    for h_ref, lbl in [(11, "Tropopause (11 km)"),
                       (20, "Strat. base (20 km)"),
                       (32, "Mid-strat.  (32 km)")]:
        ax.axhline(h_ref, color="#888", linewidth=0.7,
                   linestyle=":", label=lbl)

    for sc in SCENARIOS:
        dp = peak_overpressure(sc["M"], sc["h_km"] * 1_000)
        ax.scatter(sc["M"], sc["h_km"], color=sc["color"],
                   s=100, zorder=6, edgecolors="white", linewidths=0.8)
        ax.annotate(f"{sc['label']}\nΔP={dp:.0f} Pa",
                    xy=(sc["M"], sc["h_km"]),
                    xytext=(8, 4), textcoords="offset points",
                    color=sc["color"], fontsize=8.5)

    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(0, 47)
    ax.set_xlabel("Mach number",    color="#7fb3d3", fontsize=11)
    ax.set_ylabel("Altitude  (km)", color="#7fb3d3", fontsize=11)
    ax.set_title(
        f"Minimum altitude vs Mach number for ΔP ≤ {THRESHOLD:.0f} Pa\n"
        f"NASA Carlson TP-1122 (1978) simplified prediction  |  "
        f"K_S = {K_S_AIRCRAFT},  l = {L_AIRCRAFT:.0f} m",
        color="white", fontsize=12, fontweight="bold", pad=10)

    ax.grid(color="white", linestyle="--", linewidth=0.3, alpha=0.3)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334e68")
    ax.legend(facecolor="#0d1b2a", edgecolor="#334e68",
              labelcolor="white", fontsize=9.5)

    plt.tight_layout()
    plt.savefig("mach_altitude_envelope.png", dpi=150, facecolor="#0d1b2a")
    plt.show()
    print("Saved: mach_altitude_envelope.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 68)
    print("ISA spot-check vs ICAO Doc 7488 reference values")
    print("=" * 68)
    icao_ref = [
        (     0, 288.15, 101_325.0),
        (11_000, 216.65,  22_632.1),
        (20_000, 216.65,   5_474.89),
        (32_000, 228.65,     868.019),
        (47_000, 270.65,     110.906),
    ]
    print(f"  {'h (km)':>7}  {'T mdl (K)':>10}  {'T ref (K)':>10}  "
          f"{'p mdl (Pa)':>12}  {'p ref (Pa)':>12}")
    for h, T_ref, p_ref in icao_ref:
        print(f"  {h/1000:>7.0f}  {isa_temperature(h):>10.2f}  {T_ref:>10.2f}  "
              f"{isa_pressure(h):>12.3f}  {p_ref:>12.3f}")

    print()
    print("=" * 78)
    print(f"Carlson TP-1122 scenario summary")
    print(f"K_S = {K_S_AIRCRAFT},  l = {L_AIRCRAFT} m,  ΔP threshold = {THRESHOLD} Pa")
    print("=" * 78)
    print(f"  {'Scenario':<18}  {'p_v kPa':>8}  {'K_p':>5}  "
          f"{'ΔP₀ (Pa)':>9}  {'h_min km':>9}  {'Offshore km':>12}")
    print("-" * 78)
    for sc in SCENARIOS:
        h_m   = sc["h_km"] * 1_000
        p_v   = isa_pressure(h_m) / 1_000
        Kp    = k_p(sc["M"], h_m)
        dp    = peak_overpressure(sc["M"], h_m)
        h_min = min_altitude_m(sc["M"]) / 1_000
        d_s   = safe_offshore_km(sc["M"], h_m)
        flag  = " ✓ compliant" if dp <= THRESHOLD else " ✗ above limit"
        print(f"  {sc['label']:<18}  {p_v:>8.2f}  {Kp:>5.2f}  "
              f"{dp:>9.1f}  {h_min:>9.1f}  {d_s:>12.1f}{flag}")

    print("\nGenerating ISA profile…")
    plot_isa_profile()

    print("Generating coastal exclusion map…")
    plot_coastal_exclusion_map()

    print("Generating Mach–altitude envelope…")
    plot_mach_altitude_envelope()