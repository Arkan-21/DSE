
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
import matplotlib.pyplot as plt

def plot_kw_vs_tau():
    x1 = np.linspace(0.02, 0.2, 100)
    y1 = []
    for i in range(len(x1)):
        tau = x1[i]
        y1a = 1053.3 * tau**4- 863.55* tau**3 + 243.06* tau**2 - 22.026 * tau + 3.3655 #waverider
        y1.append(y1a)

    x2 = np.linspace(0.02, 0.3, 100)
    y2 = []
    for i in range(len(x2)): 
        tau = x2[i]
        y2a = 1604.6 * tau**4 - 936.96 * tau**3 + 203.15 * tau**2 - 15.014* tau + 2.9932 #wing_body
        y2.append(y2a)

    x3 = np.linspace(0.02, 0.3, 100)
    y3 = []
    for i in range(len(x3)):
        tau = x3[i]
        y3a = -326 * tau**4 + 189.99* tau**3 -22.369* tau**2 + 3.7082* tau+ 2.3081 #blended_body
        y3.append(y3a)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x1, y1, color='#D85A30', linewidth=2.0, label='Waverider')
    ax.plot(x2, y2, color='#378ADD', linewidth=2.0, label='Wing body')
    ax.plot(x3, y3, color='#1D9E75', linewidth=2.0, label='Blended body')

    ax.set_xlabel(r"Küchemann's $\tau$", fontsize=12)
    ax.set_ylabel(r'$K_W$', fontsize=12)
    ax.set_title(r'$K_W$ vs $\tau$', fontsize=13)

    ax.set_xlim(0, 0.4)
    ax.tick_params(direction='in', which='both', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    ax.legend(frameon=True, fontsize=10, edgecolor='grey')

    plt.tight_layout()
    plt.show()


def plot_TOGW_vs_tau():

    # x1 = [0.08, 0.1, 0.12 , 0.14, 0.16]        # Waverider
    # y1 = [122837.843, 114543.786,  109618.666,  106395.507,  104039.772]

    # x2 = [0.08, 0.1, 0.12 , 0.14, 0.16]         # Wing body
    # y2 = [120671.326, 112588.843, 107356.857, 103627.970, 100783.239]
  
    # x3 = [0.08, 0.1, 0.12 , 0.14, 0.16]      # Blended body
    # y3 = [117193.469, 109654.585, 104724.22,  101378.720, 99062.630]

    x = 0.12
    y1 = 109618.666
    y2 = 107356.857
    y3 = 104724.22

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [
        ('Waverider',     y1, '#D85A30', '^'),
        ('Wing body',     y2, '#378ADD', 'o'),
        ('Blended body',  y3, '#1D9E75', 's'),
    ]

    for label, y, color, marker in configs:
        ax.scatter(x, y, color=color, marker=marker, s=100, zorder=3, label=label)
        ax.annotate(
            f'{y:,.0f} kg',
            xy=(x, y),
            xytext=(8, 6),
            textcoords='offset points',
            fontsize=9,
            color=color,
        )

    ax.set_xlabel(r"Küchemann's $\tau$", fontsize=12)
    ax.set_ylabel(r'TOGW [kg]', fontsize=12)
    ax.set_title(r'TOGW vs $\tau$  at $\tau = 0.12$', fontsize=13)

    ax.set_xlim(0.05, 0.2)
    ax.set_ylim(100_000, 115_000)
    ax.tick_params(direction='in', which='both', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_TOGW_vs_payload():
    x1 = np.linspace(4000, 10000, 7)        # Waverider
    y1 = [ 98641.839 , 100687.041, 102709.754, 104724.22, 106730.791 ,108729.881, 110721.833 ]
    for i in range(len(y1)):
        y1[i] = (y1[i]-104724.22)/ 104724.22 *100 

    y2 = [414.423, 420.327, 426.038, 431.693, 437.294, 442.843, 448.341 ]
    for i in range(len(y2)):
        y2[i] = (y2[i]-431.693)/ 431.693 *100

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x1, y1, color='#D85A30', linestyle = 'solid', linewidth=2.0, label='TOGW')
    ax.plot(x1, y2, color='#378ADD', linestyle = 'solid' ,linewidth=2.0, label=r'S$_{plan}$')

    ax.set_xlabel(r"Payload Mass M$_{pay}$", fontsize=12)
    ax.set_ylabel(r'Change of output [%]', fontsize=12)
    ax.set_title(r'TOGW and S$_{plan}$ vs M$_{pay}$', fontsize=13)

    ax.set_xlim(3500, 10500.)
    ax.tick_params(direction='in', which='both', labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    ax.legend(frameon=True, fontsize=10, edgecolor='grey')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    #plot_kw_vs_tau()
    #plot_TOGW_vs_tau()
    plot_TOGW_vs_payload()