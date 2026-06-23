"""
Visualise the initial geometry and verify the shock + structures constraints
exactly as implemented in geometry_generator.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from geometry_generator import BASELINE_PARAMS, shock_constraint, structures_feasibility_constraint

p = BASELINE_PARAMS.copy()

TOTAL_LENGTH     = 60.5
FUNCTIONAL_LENGTH = 43.0

# ── wing planform from BASELINE_PARAMS ────────────────────────────────────────
wing_x = p["wing_x_location"] * TOTAL_LENGTH          # root LE x  (= 12.0 m)

con_chord_abs = p["con_chord"] * TOTAL_LENGTH          # connector chord
c0 = p["chord0"] * con_chord_abs
c1 = p["chord1"] * c0
c2 = p["chord2"] * c1
c3 = p["chord3"] * c2

# half-span stations (y = 0 at centreline)
y_sta = np.array([
    0.0,
    p["con_span"],
    p["con_span"] + p["span1"],
    p["con_span"] + p["span1"] + p["span2"],
    p["con_span"] + p["span1"] + p["span2"] + p["span3"],
])
half_span = y_sta[-1]

# leading-edge x at each station
x_le = wing_x + np.cumsum([
    0.0,
    p["con_span"] * np.tan(np.radians(p["con_sweep"] * 90)),
    p["span1"]    * np.tan(np.radians(p["sweep1"]    * 90)),
    p["span2"]    * np.tan(np.radians(p["sweep2"]    * 90)),
    p["span3"]    * np.tan(np.radians(p["sweep3"]    * 90)),
])

# trailing-edge x at each station (LE + local chord)
x_te = x_le + np.array([con_chord_abs, c0, c1, c2, c3])

# ── evaluate both constraints ─────────────────────────────────────────────────
sc  = shock_constraint(**p)
stc = structures_feasibility_constraint(**p)

# ── shock cone (reproduced from shock_constraint source) ──────────────────────
# x_tip += start_of_shock  =>  cone boundary: y = (x_fuselage + 2.7) * tan(16.26°)
# i.e. effective cone origin is at x = -2.7 m in fuselage coordinates
SHOCK_ORIGIN_X = -2.7
SHOCK_ANGLE    = np.radians(16.26)
shock_r_at_tip = (x_le[-1] - SHOCK_ORIGIN_X) * np.tan(SHOCK_ANGLE)   # cone radius at tip LE x

# ── tip TE as used in structures_feasibility_constraint (uses FUNCTIONAL_LENGTH) ─
tip_te_constr = (x_le[-1]
                 + p["chord3"] * p["chord2"] * p["chord1"] * p["chord0"]
                 * p["con_chord"] * FUNCTIONAL_LENGTH)
tip_te_actual = x_te[-1]   # based on TOTAL_LENGTH (what generate_geometry actually builds)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10))

# fuselage outline
CABIN_HW = 4.0 / 2
ax.fill([0, TOTAL_LENGTH, TOTAL_LENGTH, 0],
        [0, 0, CABIN_HW, CABIN_HW],
        color='#BBBBBB', alpha=0.5, label='Fuselage (half-width)')
ax.plot([0, TOTAL_LENGTH, TOTAL_LENGTH, 0, 0],
        [0, 0, CABIN_HW, CABIN_HW, 0], 'k-', lw=1.5)

# shock cone boundary  y = (x + 2.7) * tan(16.26°)
x_cone = np.linspace(SHOCK_ORIGIN_X, TOTAL_LENGTH + 3, 400)
y_cone = (x_cone - SHOCK_ORIGIN_X) * np.tan(SHOCK_ANGLE)
ax.plot(x_cone, y_cone, 'r-', lw=2.5,
        label=f'Mach shock boundary  (16.26°, effective origin x = {SHOCK_ORIGIN_X} m)')
ax.plot(SHOCK_ORIGIN_X, 0, 'r|', ms=14, markeredgewidth=2, zorder=6)

# wing planform — four sections, one half
SECTION_COLORS  = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
SECTION_LABELS  = ['Connector', 'Section 1', 'Section 2', 'Section 3']
for i in range(len(y_sta) - 1):
    xs = [x_le[i], x_le[i+1], x_te[i+1], x_te[i]]
    ys = [y_sta[i], y_sta[i+1], y_sta[i+1], y_sta[i]]
    ax.fill(xs, ys, alpha=0.35, color=SECTION_COLORS[i], label=SECTION_LABELS[i])
    ax.plot(xs + [xs[0]], ys + [ys[0]], '-', color=SECTION_COLORS[i], lw=1.2)

ax.plot(x_le, y_sta, 'b-', lw=2.5)
ax.plot(x_te, y_sta, 'b--', lw=1.5)

# wing tip LE
ax.plot(x_le[-1], half_span, 'g^', ms=14, zorder=7, markeredgecolor='k',
        label=f'Wing tip LE  x = {x_le[-1]:.2f} m,  half-span = {half_span:.2f} m')

# ── shock constraint annotation ────────────────────────────────────────────────
# vertical arrow from half_span to shock radius at that x
ax.plot([x_le[-1], x_le[-1]], [half_span, shock_r_at_tip],
        color='red', lw=1.5, ls=':')
ax.annotate('', xy=(x_le[-1], shock_r_at_tip), xytext=(x_le[-1], half_span),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2.0, mutation_scale=14))
ax.plot(x_le[-1], shock_r_at_tip, 'rx', ms=12, zorder=7, markeredgewidth=2,
        label=f'Shock boundary at tip x  y = {shock_r_at_tip:.2f} m')

shock_ok = sc >= 0
ax.text(x_le[-1] + 0.7, 0.5 * (shock_r_at_tip + half_span),
        f"shock margin = {sc:+.2f} m\n{'✓ OK' if shock_ok else '✗ VIOLATED'}",
        color='darkgreen' if shock_ok else 'darkred',
        fontsize=10, va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# ── structures constraint annotation ──────────────────────────────────────────
ax.axvline(TOTAL_LENGTH, color='#2ca02c', lw=2.5, ls='--',
           label=f'Fuselage end  x = {TOTAL_LENGTH} m')

y_ann = half_span + 1.8
ax.annotate('', xy=(tip_te_constr, y_ann), xytext=(TOTAL_LENGTH, y_ann),
            arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=2.0, mutation_scale=14))

struct_ok = stc >= 0
ax.text(0.5 * (tip_te_constr + TOTAL_LENGTH), y_ann + 0.7,
        f"structures margin = {stc:+.2f} m  {'✓ OK' if struct_ok else '✗ VIOLATED'}",
        ha='center', color='darkgreen' if struct_ok else 'darkred',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

ax.plot(tip_te_constr, y_sta[-1], 'gs', ms=12, zorder=7, markeredgecolor='k',
        label=f'Wing tip TE (constraint formula, f_len={FUNCTIONAL_LENGTH})  x = {tip_te_constr:.2f} m')
ax.plot(tip_te_actual, y_sta[-1] - 0.4, 'bs', ms=10, zorder=7, markeredgecolor='k',
        label=f'Wing tip TE (geometry formula, tot_len={TOTAL_LENGTH})  x = {tip_te_actual:.2f} m')

ax.set_xlim(-5, TOTAL_LENGTH + 6)
ax.set_ylim(-1, half_span + 5)
ax.set_xlabel('x — fuselage axis [m]', fontsize=12)
ax.set_ylabel('y — half-span [m]', fontsize=12)
ax.set_title(
    'Constraint verification — baseline geometry\n'
    f'shock_constraint = {sc:+.3f} m   |   structures_constraint = {stc:+.3f} m',
    fontsize=13
)
ax.legend(loc='upper left', fontsize=8, framealpha=0.93, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('constraint_verification.png', dpi=150)
plt.show()

print(f"\n{'='*55}")
print(f"  shock_constraint      = {sc:+.4f} m"
      f"  ({'OK' if sc >= 0 else 'VIOLATED'})")
print(f"  structures_constraint = {stc:+.4f} m"
      f"  ({'OK' if stc >= 0 else 'VIOLATED'})")
print(f"{'='*55}")
print(f"  Wing root LE x        = {wing_x:.2f} m")
print(f"  Wing tip  LE x        = {x_le[-1]:.2f} m")
print(f"  Half-span             = {half_span:.2f} m")
print(f"  Shock radius at tip x = {shock_r_at_tip:.2f} m")
print(f"  Wing tip TE (constr)  = {tip_te_constr:.2f} m  "
      f"(uses functional_length={FUNCTIONAL_LENGTH})")
print(f"  Wing tip TE (geom)    = {tip_te_actual:.2f} m  "
      f"(uses total_length={TOTAL_LENGTH})")
print(f"{'='*55}\n")
