"""
coastal_zones.py
----------------
Maps Mach-speed sonic boom exclusion zones from coastlines and plots
multiple AMS→LAX hypersonic route options on a polar-safe projection.

Zones represent the minimum distance from the coastline (over ocean)
that an aircraft must fly to avoid its sonic boom reaching the coast.

Dependencies:
    conda install geopandas shapely matplotlib pyproj cartopy geodatasets
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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import Transformer, Geod
from geodatasets import get_path

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — edit these values
# ---------------------------------------------------------------------------

# Zone distances from coastline in km — replace with outputs from your other program
# Each zone = minimum ocean distance to avoid sonic boom at that Mach number
ZONE_DISTANCES_KM = {
    "Mach 1": 50,
    "Mach 2": 150,
    "Mach 3": 300,
    "Mach 4": 500,
    "Mach 5": 750,
}

ZONE_COLORS = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#c77dff"]
ZONE_ALPHA  = 0.4

# Airport coordinates (longitude, latitude)
AMS = (4.764,    52.308)
LAX = (-118.408, 33.943)

# Flight routes — (display name, line colour, list of (lon, lat) waypoints)
# Cartopy draws these as proper great circle segments automatically
ROUTES = [
    ("North Atlantic",  "#00d4ff", [AMS, (-20, 58), (-45, 55), (-80, 45), LAX]),
    ("Polar",           "#ff9f1c", [AMS, (-169, 65), (-170, 50), LAX]),
    ("Greenland",       "#ff4dff", [AMS, LAX]),
]

# ---------------------------------------------------------------------------
# 2. LOAD NATURAL EARTH DATA
# ---------------------------------------------------------------------------

def load_natural_earth():
    """Load global land polygons from Natural Earth via geodatasets."""
    world = gpd.read_file(get_path("naturalearth.land"))
    return world

# ---------------------------------------------------------------------------
# 3. BUILD OCEAN EXCLUSION ZONES
# ---------------------------------------------------------------------------

def project_geometry(geom, from_crs="EPSG:4326", to_crs="EPSG:3857"):
    import shapely.ops as sops
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return sops.transform(transformer.transform, geom)

def unproject_geometry(geom, from_crs="EPSG:3857", to_crs="EPSG:4326"):
    import shapely.ops as sops
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return sops.transform(transformer.transform, geom)

def build_ocean_zones(land_union, distances_km):
    """
    Build concentric ocean-only exclusion zones at given distances from coast.
    Returns list of Shapely geometries in WGS84, index 0 = closest to coast.
    """
    # Project to metres for buffering (Web Mercator, good for mid-latitudes)
    land_proj  = project_geometry(land_union)
    global_box = project_geometry(box(-180, -85, 180, 85))  # Mercator clips ~85°
    ocean_proj = global_box.difference(land_proj)

    zones      = []
    prev_buf   = land_proj

    for dist_km in distances_km:
        buf        = land_proj.buffer(dist_km * 1000)
        ring       = buf.difference(prev_buf)
        ring_ocean = ring.intersection(ocean_proj)
        zones.append(unproject_geometry(ring_ocean))
        prev_buf   = buf

    return zones

# ---------------------------------------------------------------------------
# 4. GREAT CIRCLE ROUTES
# ---------------------------------------------------------------------------

def great_circle_segment(lon1, lat1, lon2, lat2, n=80):
    """Return n+2 points along the geodesic between two coordinates."""
    geod   = Geod(ellps="WGS84")
    pts    = geod.npts(lon1, lat1, lon2, lat2, n)
    lons   = [lon1] + [p[0] for p in pts] + [lon2]
    lats   = [lat1] + [p[1] for p in pts] + [lat2]
    return lons, lats

def build_route(waypoints, n_per_segment=60):
    """Stitch great circle segments between all waypoints."""
    all_lons, all_lats = [], []
    for i in range(len(waypoints) - 1):
        lons, lats = great_circle_segment(*waypoints[i], *waypoints[i + 1], n_per_segment)
        if i > 0:                          # trim duplicate junction point
            lons, lats = lons[1:], lats[1:]
        all_lons.extend(lons)
        all_lats.extend(lats)
    return all_lons, all_lats

# ---------------------------------------------------------------------------
# 5. PLOTTING
# ---------------------------------------------------------------------------

def plot_map(world, zones, routes):
    # North Polar Stereographic — handles polar routes without distortion
    proj = ccrs.NorthPolarStereo(central_longitude=-60)
    geo  = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=(13, 13),
        subplot_kw={"projection": proj},
        facecolor="#0d1b2a",
    )
    ax.set_facecolor("#1a3a5c")

    # Show from ~25°N to the pole so LAX and AMS are both visible
    ax.set_extent([-180, 180, 25, 90], crs=geo)

    # --- Base map features ---
    ax.add_feature(cfeature.OCEAN,     color="#1a3a5c", zorder=1)
    ax.add_feature(cfeature.LAND,      color="#3d5a3e", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#a8c5a0", zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#556677", zorder=3)
    gl = ax.gridlines(color="white", linestyle="--", linewidth=0.3, alpha=0.25, zorder=1)

    # --- Exclusion zones ---
    zone_names = list(ZONE_DISTANCES_KM.keys())
    zone_patches = []
    for zone_geom, color, name in zip(reversed(zones),
                                       reversed(ZONE_COLORS),
                                       reversed(zone_names)):
        gdf = gpd.GeoDataFrame(geometry=[zone_geom], crs="EPSG:4326")
        gdf.plot(ax=ax, color=color, alpha=ZONE_ALPHA,
                 edgecolor="none", transform=geo, zorder=2)

    # Build legend patches in correct order (Mach 1 first)
    for color, name in zip(ZONE_COLORS, zone_names):
        zone_patches.append(
            mpatches.Patch(color=color, alpha=0.7,
                           label=f"{name}  ({ZONE_DISTANCES_KM[name]} km)")
        )

    # --- Flight routes ---
    route_lines = []
    for name, color, waypoints in routes:
        lons, lats = build_route(waypoints)
        ax.plot(lons, lats,
                color=color, linewidth=2.2, linestyle="-",
                transform=ccrs.Geodetic(), zorder=5)
        route_lines.append(
            mlines.Line2D([], [], color=color, linewidth=2.2, label=name)
        )

    # --- Airport markers ---
    for label, (lon, lat) in [("AMS", AMS), ("LAX", LAX)]:
        ax.plot(lon, lat, "o",
                color="white", markersize=7, markeredgecolor="#0d1b2a",
                transform=geo, zorder=6)
        ax.text(lon + 3, lat - 1, label,
                color="white", fontsize=10, fontweight="bold",
                transform=geo, zorder=6)

    # --- Legend ---
    base_patches = [
        mpatches.Patch(color="#3d5a3e", label="Land"),
        mpatches.Patch(color="#1a3a5c", label="Ocean"),
    ]
    ax.legend(
        handles=base_patches + zone_patches + route_lines,
        loc="lower left",
        facecolor="#0d1b2a", edgecolor="#334e68",
        labelcolor="white", fontsize=9,
        title="Legend", title_fontsize=10,
    )
    ax.get_legend().get_title().set_color("white")

    ax.set_title(
        "AMS → LAX Hypersonic Route Options\nSonic Boom Exclusion Zones by Mach Number",
        color="white", fontsize=14, pad=14, fontweight="bold",
    )

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# 6. POINT CLASSIFIER (for later use with trajectory program)
# ---------------------------------------------------------------------------

def classify_point(lon, lat, land_union, zones):
    """
    Return the zone label for a (lon, lat) coordinate.
    Useful when your trajectory program needs to check route compliance.

    Returns: "land" | "zone_Mach1" | ... | "zone_Mach5" | "open_ocean"
    """
    from shapely.geometry import Point
    pt         = Point(lon, lat)
    zone_names = list(ZONE_DISTANCES_KM.keys())

    if land_union.contains(pt):
        return "land"
    for name, zone_geom in zip(zone_names, zones):
        if zone_geom.contains(pt):
            return f"zone_{name.replace(' ', '')}"
    return "open_ocean"

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading Natural Earth data…")
    world      = load_natural_earth()
    land_union = unary_union(world.geometry)

    print(f"Building {len(ZONE_DISTANCES_KM)} ocean exclusion zones…")
    zones = build_ocean_zones(land_union, list(ZONE_DISTANCES_KM.values()))

    print("Plotting…")
    plot_map(world, zones, ROUTES)

    # Quick demo of point classifier
    test_points = [
        (4.9,   52.3, "Amsterdam"),
        (-30.0, 60.0, "North Atlantic"),
        (0.0,   88.0, "North Pole"),
        (-118.4, 33.9, "Los Angeles"),
    ]
    print("\nPoint classification:")
    for lon, lat, name in test_points:
        result = classify_point(lon, lat, land_union, zones)
        print(f"  {name:25s} → {result}")