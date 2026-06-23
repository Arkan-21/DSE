"""fuselage_cross_section.py
Cabin cross-section of the Mach 5 hypersonic vehicle.
Geometry from Geogebra (1 unit = 10 mm → scale × 0.01 m/unit).

Colours:
  Red    – aisle
  Yellow – headroom
  Orange – overhead storage
  Blue   – seats
  Green  – underdeck cargo
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Polygon

plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})

# ── Fuselage geometry (metres) ─────────────────────────────────────────────
# Geogebra: centre (0, 57.5), R_outer=200, R_inner²=33931.25  (units × 0.01 = m)
cx, cy = 0.0, 0.575
R_out  = 2.000
R_in   = np.sqrt(33931.25) * 0.01   # ≈ 1.842 m

# ── Key heights (z) [m] ────────────────────────────────────────────────────
Z0    = 0.00   # cabin floor
Z_s   = 1.15   # top of seat backs
Z_h   = 1.65   # bottom of overhead bins
Z_top = 2.30   # ceiling / top of overhead bins

# ── Lateral bounds (y) [m] ────────────────────────────────────────────────
al, ar = -0.25,  0.25    # aisle
sl, sr = -1.75,  1.75   # seat outer edges

# Overhead trapezoid: (-1.50,1.65)→(1.50,1.65)→(0.25,2.30)→(-0.25,2.30)
OH = [(-1.50, Z_h), (1.50, Z_h), (ar, Z_top), (al, Z_top)]

# ── Underdeck cargo [m] ────────────────────────────────────────────────────
cgl, cgr, cgb = -0.955, 0.955, -1.00

# ── Seat dividers: 3 seats × 0.5 m per side ──────────────────────────────
left_divs  = [sl + 0.50, sl + 1.00]   # −1.25, −0.75
right_divs = [ar + 0.50, ar + 1.00]   #  0.75,  1.25

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_aspect('equal')

# Invisible clip patch (inner fuselage circle) — added first for reference
clip = Circle((cx, cy), R_in, fc='none', ec='none', zorder=0)
ax.add_patch(clip)

def cp(patch):
    ax.add_patch(patch)
    patch.set_clip_path(clip)
    return patch

def cl(xs, ys, **kw):
    for line in ax.plot(xs, ys, **kw):
        line.set_clip_path(clip)

# 1. Fuselage shell: outer grey → inner white
ax.add_patch(Circle((cx, cy), R_out, fc='#b0bec5', ec='#263238', lw=3.0, zorder=1))
ax.add_patch(Circle((cx, cy), R_in,  fc='white',   ec='#263238', lw=2.0, zorder=2))

# 2. GREEN – underdeck cargo
cp(Rectangle((cgl, cgb), cgr - cgl, Z0 - cgb,
             fc='#a5d6a7', ec='#2e7d32', lw=2.0, zorder=3))

# 3. BLUE – seats (left and right of aisle)
for x0, w in [(sl, al - sl), (ar, sr - ar)]:
    cp(Rectangle((x0, Z0), w, Z_s - Z0,
                 fc='#90caf9', ec='#1565c0', lw=2.0, zorder=3))

for xd in left_divs:
    cl([xd, xd], [Z0, Z_s], color='#1565c0', lw=1.2, zorder=4)
for xd in right_divs:
    cl([xd, xd], [Z0, Z_s], color='#1565c0', lw=1.2, zorder=4)

# 4. YELLOW – headroom (above seats, below overhead bins)
for x0, w in [(sl, al - sl), (ar, sr - ar)]:
    cp(Rectangle((x0, Z_s), w, Z_h - Z_s,
                 fc='#fff59d', ec='#f9a825', lw=2.0, zorder=3))

# 5. ORANGE – overhead storage (trapezoid)
cp(Polygon(OH, closed=True, fc='#ffcc80', ec='#e65100', lw=2.0, zorder=3))

# 6. RED – aisle column, full height (on top)
cp(Rectangle((al, Z0), ar - al, Z_top - Z0,
             fc='#ef9a9a', ec='#c62828', lw=2.0, zorder=5))

# 7. Floor line & cargo floor
ax.plot([sl, sr],   [Z0,  Z0],  'k-', lw=2.5, zorder=6)
ax.plot([cgl, cgr], [cgb, cgb], color='#2e7d32', lw=1.8, ls='--', zorder=6)

# 8. Fuselage outlines redrawn on top
ax.add_patch(Circle((cx, cy), R_out, fc='none', ec='#263238', lw=3.0, zorder=9))
ax.add_patch(Circle((cx, cy), R_in,  fc='none', ec='#263238', lw=2.0, zorder=9))

# ── Region labels ──────────────────────────────────────────────────────────
fs = 13

ax.text(0, (Z0 + Z_top) / 2, 'Aisle\n(0.5 m)',
        ha='center', va='center', fontsize=fs, fontweight='bold',
        color='#7b0000', zorder=10)

for xm in [(sl + al) / 2, (ar + sr) / 2]:
    ax.text(xm, (Z0 + Z_s) / 2, 'Seats\n3 × 0.5 m',
            ha='center', va='center', fontsize=fs, fontweight='bold',
            color='#0d47a1', zorder=10)
    ax.text(xm, (Z_s + Z_h) / 2, 'Head-\nroom',
            ha='center', va='center', fontsize=fs - 2, color='#5d4037', zorder=10)

ax.text(0, (Z_h + Z_top) / 2, 'Overhead\nstorage',
        ha='center', va='center', fontsize=fs, fontweight='bold',
        color='#bf360c', zorder=10)

ax.text((cgl + cgr) / 2, (cgb + Z0) / 2, 'Underdeck cargo',
        ha='center', va='center', fontsize=fs, fontweight='bold',
        color='#1b5e20', zorder=10)

# ── Dimension callouts ─────────────────────────────────────────────────────
arr = dict(arrowprops=dict(arrowstyle='<->', color='#222', lw=1.3),
           annotation_clip=False)

# Aisle width
ax.annotate('', xy=(ar, -0.20), xytext=(al, -0.20), **arr)
ax.text(0, -0.29, '0.5 m', ha='center', va='top', fontsize=13, color='#333')

# Left seat width
ax.annotate('', xy=(al, -0.20), xytext=(sl, -0.20), **arr)
ax.text((sl + al) / 2, -0.29, '1.5 m', ha='center', va='top', fontsize=13, color='#333')

# Right seat width
ax.annotate('', xy=(sr, -0.20), xytext=(ar, -0.20), **arr)
ax.text((ar + sr) / 2, -0.29, '1.5 m', ha='center', va='top', fontsize=13, color='#333')




# Height markers on right
for z, lbl in [(Z_s, f'z = {Z_s} m'), (Z_h, f'z = {Z_h} m'), (Z_top, f'z = {Z_top} m')]:
    ax.plot([sr + 0.05, sr + 0.25], [z, z], color='#555', lw=1.0)
    ax.text(sr + 0.28, z, lbl, va='center', fontsize=13, color='#444')



# Skin / TPS thickness note
theta = np.radians(50)
xi = R_in  * np.cos(theta)
zi = cy + R_in  * np.sin(theta)
xo = R_out * np.cos(theta)
zo = cy + R_out * np.sin(theta)
ax.annotate(
    f'Structure + TPS\n({(R_out - R_in) * 1000:.0f} mm)',
    xy=((xi + xo) / 2, (zi + zo) / 2),
    xytext=(xi + 0.28, zi + 0.42),
    fontsize=12, color='#37474f', ha='left',
    arrowprops=dict(arrowstyle='->', color='#37474f', lw=1.0),
    annotation_clip=False, zorder=11)

# ── Legend ─────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(fc='#ef9a9a', ec='#c62828', lw=1.5, label='Aisle'),
    mpatches.Patch(fc='#fff59d', ec='#f9a825', lw=1.5, label='Headroom'),
    mpatches.Patch(fc='#ffcc80', ec='#e65100', lw=1.5, label='Overhead storage'),
    mpatches.Patch(fc='#90caf9', ec='#1565c0', lw=1.5, label='Seats (3 + 3)'),
    mpatches.Patch(fc='#a5d6a7', ec='#2e7d32', lw=1.5, label='Underdeck cargo'),
    mpatches.Patch(fc='#b0bec5', ec='#263238', lw=1.5, label='Structure / TPS'),
]

# Fuselage outer diameter arrow — placed above the top of the circle
z_diam = cy + R_out + 0.15
ax.annotate('', xy=(R_out, z_diam), xytext=(-R_out, z_diam), **arr)
ax.text(0, z_diam + 0.06, f'Ø {2 * R_out:.1f} m (outer)',
        ha='center', va='bottom', fontsize=13, color='#444')
# ax.legend(handles=legend_patches, loc='lower left', fontsize=13,
#           framealpha=0.95, edgecolor='#888', borderpad=0.9)

# ── Axes ───────────────────────────────────────────────────────────────────
ax.set_xlim(-2.5, 2.8)
ax.set_ylim(-1.90, 3.15)
ax.set_xlabel('Lateral (y) position  [m]', fontsize=14)
ax.set_ylabel('Vertical (z) position  [m]   (z = 0: cabin floor)', fontsize=14)
ax.set_title(
    'Fuselage Cross-Section',
    fontsize=16, fontweight='bold', pad=12)
ax.grid(True, alpha=0.18, zorder=0)
ax.axhline(0, color='k', lw=0.8, alpha=0.35, zorder=0)
ax.axvline(0, color='k', lw=0.8, alpha=0.35, zorder=0)

plt.tight_layout(pad=1.5)
plt.savefig('fuselage_cross_section.png', dpi=200, bbox_inches='tight')
plt.savefig('fuselage_cross_section.pdf', dpi=300, bbox_inches='tight')
plt.show()
print('Saved -> fuselage_cross_section.png / .pdf')
