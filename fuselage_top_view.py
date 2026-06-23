"""fuselage_top_view.py — Top-view layout of the Mach 5 hypersonic vehicle.

Nose  : circular arc from fairing_calculator.py (R = 2 m, spans 0–10.5 m)
Body  : Ø 4 m cylinder (R = 2 m)
Tail  : Sears–Haack body from fairing_calculator.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Parameters from fairing_calculator.py ──────────────────────────────────
R        = 2.0
R_to_L   = 0.1906
L_to_rho = 0.3679

L_nose_f = R / R_to_L            # formula nose length  ~10.49 m
rho_nose  = L_nose_f / L_to_rho   # circular-arc radius  ~28.52 m

# ── Section boundaries [m] ─────────────────────────────────────────────────
X_NOSE   =  0.0
X_N_END  =  8.0    # colour boundary: nose cone / cockpit
X_CK_END = 10.5    # cockpit end = where nose profile meets cylinder
X_CB_END = 18.5    # cabin end
X_TK_END = 53.5    # fuel-tank end = tail fairing start
X_TAIL   = 60.5    # tail tip

L_TOT = X_TAIL
TS    = X_TK_END / L_TOT   # Sears–Haack normalised tail-start ~0.884

# ── Profile functions ──────────────────────────────────────────────────────
def nose_half_w(x):
    """Circular-arc nose cone scaled to span 0 → X_CK_END.
    x = 0 : tip (width = 0), x = X_CK_END : full cylinder (width = R)."""
    x   = np.asarray(x, float)
    x_f = L_nose_f * (1.0 - np.clip(x, X_NOSE, X_CK_END) / X_CK_END)
    return np.sqrt(np.maximum(rho_nose**2 - x_f**2, 0.0)) - (rho_nose - R)


def _vol_SH(r, L, ts):
    return (r * np.pi / 8.0 / (ts * (1 - ts))**0.75)**2 * 1.5 * L


_V = _vol_SH(R, L_TOT, TS)


def tail_half_w(x):
    """Sears–Haack tail: x = X_TK_END → R, x = X_TAIL → 0."""
    x  = np.asarray(x, float)
    xn = np.clip(x, X_TK_END, X_TAIL) / L_TOT
    return (8.0 / np.pi) * np.sqrt(2.0 * _V / (3.0 * L_TOT)) * \
           np.maximum(xn * (1 - xn), 0.0)**0.75


def half_w(x):
    x = np.asarray(x, float)
    return np.where(x <= X_CK_END, nose_half_w(x),
           np.where(x >= X_TK_END, tail_half_w(x),
                    np.full_like(x, R)))


# ── Continuous outline ─────────────────────────────────────────────────────
x_out = np.concatenate([
    np.linspace(X_NOSE,   X_CK_END, 800),
    np.linspace(X_CK_END, X_TK_END,  40),
    np.linspace(X_TK_END, X_TAIL,   600),
])
w_out = half_w(x_out)

# ── Figure — A4 landscape, no equal-aspect (15:1 ratio would make it unreadable) ──
fig, ax = plt.subplots(figsize=(16.54, 6.0))   # A4 landscape in inches

sections = [
    (X_NOSE,   X_N_END,  '#aed6f1', 'Nose cone\n0 – 8 m'),
    (X_N_END,  X_CK_END, '#f9e79f', 'Cockpit\n8 – 10.5 m'),
    (X_CK_END, X_CB_END, '#a9dfbf', 'Passenger cabin\n10.5 – 18.5 m'),
    (X_CB_END, X_TK_END, '#fdebd0', 'Fuel tanks\n18.5 – 53.5 m'),
    (X_TK_END, X_TAIL,   '#d2b4de', 'Tail fairing\n53.5 – 60.5 m'),
]

# Section fills
for x0, x1, clr, _ in sections:
    xs = np.linspace(x0, x1, 600)
    ws = half_w(xs)
    ax.fill_between(xs, -ws, ws, color=clr, alpha=0.85, zorder=1)

# Outer profile
ax.plot(x_out,  w_out, 'k-', lw=2.2, zorder=5)
ax.plot(x_out, -w_out, 'k-', lw=2.2, zorder=5)

# Section dividers (x=8 divides nose cone / cockpit colouring within the nose shape)
for xd in (X_N_END, X_CK_END, X_CB_END, X_TK_END):
    wd = float(half_w(np.array([xd]))[0])
    ax.plot([xd, xd], [-wd, wd], '--', color='#555', lw=1.3, zorder=4)

# ── Door: port side (top of drawing), 1 m wide, at x = 10.5 m ────────────
DOOR_W, DOOR_T = 0.76, 0.22
ax.add_patch(mpatches.Rectangle(
    (X_CK_END, R - DOOR_T), DOOR_W, DOOR_T,
    lw=1.8, edgecolor='#c0392b', facecolor='#f1948a', zorder=7))

# ----- Emeergency exit, 0.6 m wide, other si
DOOR_W, DOOR_T = 0.6, 0.12
ax.add_patch(mpatches.Rectangle(
    (X_CK_END + 0.5*(X_CB_END - X_CK_END) - 0.5*DOOR_W, -R ), DOOR_W, DOOR_T,
    lw=1.8, edgecolor='#c0392b', facecolor='#f1948a', zorder=7))
# add text label for emergency exit
ax.text(X_CK_END + 0.5*(X_CB_END - X_CK_END), -(R + DOOR_T / 2)-0.4, 'Emergency\nExit',
        ha='center', va='center', fontsize=9, color='#922b21',
        fontweight='bold', zorder=8)
# ── Lavatory: starboard side (bottom of drawing), same x-span ─────────────
LAV_W, LAV_D = 1.0, 0.9    # 1 m wide, ~0.9 m deep (fits inside cabin)
ax.add_patch(mpatches.Rectangle(
    (X_CK_END, -R + 0.05), LAV_W, LAV_D,
    lw=1.8, edgecolor='#2471a3', facecolor='#aed6f1', zorder=7))
ax.text(X_CK_END + LAV_W / 2, -R + LAV_D / 2, 'LAV',
        ha='center', va='center', fontsize=11, color='#1a5276',
        fontweight='bold', zorder=8)

# ── Entrance vestibule: transverse corridor connecting door <-> lavatory ──
# White corridor running across the full cabin width at the door/lav zone
VEST_X0, VEST_X1 = X_CK_END, X_CK_END + DOOR_W   # 10.5 – 11.5 m
ax.fill_between([VEST_X0, VEST_X1],
                -R + LAV_D + 0.05,     # above the lavatory
                 R - DOOR_T - 0.02,    # below the door strip
                color='white', alpha=0.55, zorder=3)
ax.plot([VEST_X0, VEST_X1], [ R - DOOR_T - 0.02,  R - DOOR_T - 0.02],
        ':', color='#777', lw=1.0, zorder=4)
ax.plot([VEST_X0, VEST_X1], [-R + LAV_D + 0.05, -R + LAV_D + 0.05],
        ':', color='#777', lw=1.0, zorder=4)
ax.text((VEST_X0 + VEST_X1) / 2,
        (-R + LAV_D + 0.05 + R - DOOR_T - 0.02) / 2,
        'Entrance', ha='center', va='center', fontsize=10, color='#444',
        zorder=8, rotation=90,
        bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

# ── Aisle: 0.5 m wide, centred, across passenger cabin + vestibule ────────
AW = 0.5
ax.fill_between([X_CK_END, X_CB_END], -AW / 2, AW / 2,
                color='white', alpha=0.65, zorder=4)
ax.plot([X_CK_END, X_CB_END], [ AW / 2,  AW / 2], ':', color='#777', lw=1.2, zorder=5)
ax.plot([X_CK_END, X_CB_END], [-AW / 2, -AW / 2], ':', color='#777', lw=1.2, zorder=5)
ax.text((X_CK_END + X_CB_END) / 2, 0, 'Aisle  0.5 m',
        ha='center', va='center', fontsize=12, color='#444', zorder=9,
        fontweight='bold', bbox=dict(fc='white', ec='none', alpha=0.8, pad=2))

# ── Section labels above fuselage ─────────────────────────────────────────
Y_LBL = R + 0.52
for x0, x1, clr, lbl in sections:
    xm = (x0 + x1) / 2
    if lbl == 'Cockpit\n8 – 10.5 m':
        xm -= 0.2  # shift cockpit label slightly to the left for better spacing
    if lbl == 'Passenger cabin\n10.5 – 18.5 m':
        xm += 0.8  # shift cabin label slightly to the right for better spacing
    ax.text(xm, Y_LBL, lbl, ha='center', va='bottom', fontsize=13,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=clr, alpha=0.95,
                      ec='#888', lw=0.8))

# ── Station labels below fuselage ─────────────────────────────────────────
for xd, lbl in [(X_NOSE, '0'), (X_N_END, '8'), (X_CK_END, '10.5'),
                (X_CB_END, '18.5'), (X_TK_END, '53.5'), (X_TAIL, '60.5')]:
    ax.plot([xd, xd], [-R - 0.04, -R - 0.25], 'k-', lw=0.9)
    ax.text(xd, -R - 0.30, f'{lbl} m', ha='center', va='top',
            fontsize=13, color='#222', fontweight='bold')

# ── Total-length arrow ─────────────────────────────────────────────────────
Y_ARR = -R - 1.1
ax.annotate('', xy=(X_TAIL, Y_ARR), xytext=(X_NOSE, Y_ARR),
            arrowprops=dict(arrowstyle='<->', color='#222', lw=1.3))
ax.text(L_TOT / 2, Y_ARR - 0.22,
        f'Total fuselage length: {X_TAIL:.1f} m',
        ha='center', va='top', fontsize=9.5, fontweight='bold')

# ── Axes & labels ──────────────────────────────────────────────────────────
ax.set_xlim(-1.5, X_TAIL + 1.5)
ax.set_ylim(-R - 2.1, R + 2.1)
ax.set_xlabel('Fuselage station  x  [m]', fontsize=11)
ax.set_ylabel('Lateral position  y  [m]', fontsize=11)
ax.set_title(
    'Fuselage Top View'
    r'Nose/cockpit: ogive ($\rho$ = 28.52 m, 0-10.5 m)  |  '
    'Body: diameter 4 m cylinder  |  Tail: Sears-Haack (7 m)',
    fontsize=12, fontweight='bold', pad=10)
ax.set_yticks([-2, -1, 0, 1, 2])
ax.grid(True, alpha=0.18, zorder=0)

plt.tight_layout()
plt.savefig('fuselage_top_view.pdf', dpi=150, bbox_inches='tight')
plt.show()
print("Saved -> fuselage_top_view.pdf")
