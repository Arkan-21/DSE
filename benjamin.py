import numpy as np
import matplotlib.pyplot as plt

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
