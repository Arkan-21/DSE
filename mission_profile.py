import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from isa_atmosphere import T


# ── Constants & mission parameters ───────────────────────────────────────────
g           = 9.81                              # m/s²
acc_tot     = 0.15 * g                          # m/s²
gammas      = np.arange(5, 16, 2) * np.pi / 180
M_cruise    = 5
t_cruise    = 90 * 60                           # s
total_range = 9500e3                            # m
kappa, R    = 1.4, 287.05
h_cruise    = 30000                             # m

T_cruise    = T(h_cruise)
a_cruise    = np.sqrt(kappa * R * T_cruise)    # m/s
V_cruise    = M_cruise * a_cruise              # m/s
range_cruise = t_cruise * V_cruise             # m

km = lambda x: np.asarray(x) / 1e3            # helper: metres → km

# ── Colour ramps ──────────────────────────────────────────────────────────────
n           = len(gammas)
blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, n))
red_shades  = plt.cm.Reds (np.linspace(0.4, 0.9, n))

# Marker styles for the two key events
MARKER_H   = dict(marker='^', s=70, zorder=5)   # cruise height reached
MARKER_M5  = dict(marker='*', s=120, zorder=5)  # M=5 reached

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Hypersonic Mission Profile  —  M5 Cruise at 30 km',
             fontsize=13, fontweight='bold')

# Track whether we've already added the markup labels to the legend
_legend_h_added  = False
_legend_m5_added = False

for i, gamma in enumerate(gammas):
    acc_x   = acc_tot * np.cos(gamma)
    acc_y   = acc_tot * np.sin(gamma)
    v_end_x = V_cruise * np.cos(gamma)
    v_end_y = V_cruise * np.sin(gamma)
    dx      = v_end_x**2 / (2 * acc_x)
    dh      = v_end_y**2 / (2 * acc_y)

    lbl = f'γ = {np.degrees(gamma):.0f}°'

    if dh > h_cruise:
        # Cruise height is reached before M=5
        dv_to_cruise = np.sqrt(2 * acc_y * h_cruise)
        dx_to_cruise = dv_to_cruise**2 / (2 * acc_x)
        dx_hor_acc   = (V_cruise**2 - dv_to_cruise**2) / (2 * acc_tot)

        x_h  = km(dx_to_cruise)                       # x where h_cruise reached
        x_m5 = km(dx_to_cruise + dx_hor_acc)          # x where M=5 reached

        # ── Altitude plot ──────────────────────────────────────────────────
        ax1.plot(km([0, dx_to_cruise]),
                 km([0, h_cruise]),
                 color=blue_shades[i], lw=1.8, label=lbl)
        ax1.plot(km([dx_to_cruise,
                     dx_to_cruise + dx_hor_acc + range_cruise,
                     total_range]),
                 km([h_cruise, h_cruise, 0]),
                 color=red_shades[i], lw=1.8)

        # ▲ Cruise-height marker
        mkw_h = dict(color=red_shades[i], **MARKER_H,
                     label='Cruise height reached' if not _legend_h_added else '_nolegend_')
        ax1.scatter(x_h, km(h_cruise), **mkw_h)
       
        # ★ M=5 marker
        mkw_m5 = dict(color=red_shades[i], **MARKER_M5,
                      label='M = 5 reached' if not _legend_m5_added else '_nolegend_')
        ax1.scatter(x_m5, km(h_cruise), **mkw_m5)
        
        # ── Velocity plot ──────────────────────────────────────────────────
        ax2.plot(km([0, dx_to_cruise]),
                 [0, dv_to_cruise],
                 color=blue_shades[i], lw=1.8, label=lbl)
        ax2.plot(km([dx_to_cruise,
                     dx_to_cruise + dx_hor_acc,
                     dx_to_cruise + dx_hor_acc + range_cruise,
                     total_range]),
                 [dv_to_cruise, V_cruise, V_cruise, 0],
                 color=red_shades[i], lw=1.8)

        ax2.scatter(x_h, dv_to_cruise, **dict(color=red_shades[i], **MARKER_H,
                    label='Cruise height reached' if not _legend_h_added else '_nolegend_'))
       
        ax2.scatter(x_m5, V_cruise, **dict(color=red_shades[i], **MARKER_M5,
                    label='M = 5 reached' if not _legend_m5_added else '_nolegend_'))
       
        _legend_h_added  = True
        _legend_m5_added = True

    else:
        # M=5 (and cruise height) reached simultaneously on the climb
        ax1.plot(km([0, dx]),
                 km([0, dh]),
                 color=blue_shades[i], lw=1.8, label=lbl)
        ax1.plot(km([dx, dx + range_cruise, total_range]),
                 km([dh, dh, 0]),
                 color=red_shades[i], lw=1.8)

        mkw = dict(color=red_shades[i], **MARKER_M5,
                   label='M = 5 reached' if not _legend_m5_added else '_nolegend_')
        ax1.scatter(km(dx), km(dh), **mkw)
        
        ax2.plot(km([0, dx]), [0, v_end_x], color=blue_shades[i], lw=1.8, label=lbl)
        ax2.plot(km([dx, dx + range_cruise, total_range]),
                 [v_end_x, v_end_x, 0],
                 color=red_shades[i], lw=1.8)
        ax2.scatter(km(dx), v_end_x, **dict(color=red_shades[i], **MARKER_M5,
                    label='M = 5 reached' if not _legend_m5_added else '_nolegend_'))

        _legend_m5_added = True

# ── Altitude plot: shared decorations ────────────────────────────────────────
ax1.set_xlabel('Range (km)', fontsize=11)
ax1.set_ylabel('Altitude (km)', fontsize=11)
ax1.set_title('Altitude Profile', fontsize=11)
ax1.grid(True, alpha=0.35)


# ── Velocity plot: shared decorations ────────────────────────────────────────
ax2.set_xlabel('Range (km)', fontsize=11)
ax2.set_ylabel('Velocity (m/s)', fontsize=11)
ax2.set_title('Velocity Profile', fontsize=11)
ax2.grid(True, alpha=0.35)

# Secondary Mach axis
ax2b = ax2.twinx()
ax2b.set_ylim(np.array(ax2.get_ylim()) / a_cruise)
ax2b.set_yticks([0, 1, 2, 3, 4, 5])
ax2b.set_yticklabels([f'M {m}' for m in range(6)], fontsize=8)
ax2b.set_ylabel('Mach number', fontsize=10)

# collect handles from both axes, deduplicate by label
handles, labels = [], []
seen = set()
for ax in (ax1, ax2):
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            seen.add(l)
            handles.append(h)
            labels.append(l)

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4)
plt.tight_layout()
fig.subplots_adjust(bottom=0.15)
plt.show()

