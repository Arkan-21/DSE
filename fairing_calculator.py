import numpy as np
R = 2 # 2m radius of the sausage nose
R_to_L = 0.1906
L_to_rho = 0.3679

L = R / R_to_L
rho = L / L_to_rho

def y(x, rho=rho):
    
    return np.sqrt(rho**2-x**2)

def discretize_nose(num_points=10, R=R, L=L, rho=rho):
    x = np.linspace(0, L, num_points)
    y_vals = y(x, rho=rho)
    y_rel = y_vals - (rho-R)
    slopes = circle_slope(x, R=rho)
    curvatures = circle_curvature(x, R=rho)
    return x, y_vals, y_rel, slopes, curvatures

def circle_slope(x, R=R):
    return -x / np.sqrt(R**2 - x**2)

def circle_curvature(x, R=R):
    return -R**2 / (R**2 - x**2)**(3/2)

def plot_nose(num_points=10, R=R, L=L, rho=rho, show_plot=True):
    import matplotlib.pyplot as plt

    x, _, y_rel, slopes, _ = discretize_nose(num_points=num_points, R=R, L=L, rho=rho)
    print("Slopes at discretized points:", slopes)
    # mirror for full cross-section outline
    x_cont = np.linspace(0, L, 300)
    y_cont = y(x_cont, rho=rho) - (rho - R)

    _, (ax, ax_slope) = plt.subplots(2, 1, figsize=(10, 8),
                                        gridspec_kw={'height_ratios': [2, 1]})

    # continuous reference curve (both sides)
    ax.plot(x_cont,  y_cont, 'k-', lw=1.5, label='Nose profile')
    ax.plot(x_cont, -y_cont, 'k-', lw=1.5)

    # discretized points (both sides)
    ax.plot(x,  y_rel, 'bo', ms=6, label=f'Discretized ({num_points} pts)')
    ax.plot(x, -y_rel, 'bo', ms=6)

    # vertical lines connecting upper and lower discretized points
    for xi, yi in zip(x, y_rel):
        ax.plot([xi, xi], [-yi, yi], color='steelblue', lw=0.8, ls='--', alpha=0.6)

    # tangent line segments at each discretized point (both surfaces)
    seg_len = L / (num_points + 2) * 0.3
    for xi, yi, si in zip(x, y_rel, slopes):
        # tangent unit vector
        norm = np.sqrt(1 + si**2)
        dx_seg = seg_len / norm
        dy_seg = si * seg_len / norm
        # upper surface
        ax.plot([xi - dx_seg, xi + dx_seg],
                [yi - dy_seg, yi + dy_seg],
                color='tomato', lw=1.5, alpha=0.85)
        # lower surface (slope is mirrored)
        ax.plot([xi - dx_seg, xi + dx_seg],
                [-yi + dy_seg, -yi - dy_seg],
                color='tomato', lw=1.5, alpha=0.85)

    ax.plot([], [], color='tomato', lw=1.5, label='Local slope (tangent)')
    ax.set_xlabel('x  [m]')
    ax.set_ylabel('y  [m]')
    ax.set_title(f'Discretized nose  (R={R} m, L={L:.2f} m, {num_points} points)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # slope magnitude vs x
    slopes_cont = circle_slope(x_cont, R=rho)
    ax_slope.plot(x_cont, slopes_cont, 'k-', lw=1.5, label='dy/dx (continuous)')
    ax_slope.plot(x, slopes, 'ro', ms=6, label='dy/dx (discretized)')
    ax_slope.set_xlabel('x  [m]')
    ax_slope.set_ylabel('dy/dx  (slope)')
    ax_slope.set_title('Nose surface slope')
    ax_slope.grid(True, alpha=0.3)
    ax_slope.legend()

    plt.tight_layout()
    if show_plot:
        plt.show()

    cross_sec_d = 2 * y_rel
    print("Discretized cross-section diameters at x-locations:")
    for xi, di in zip(x, cross_sec_d):
        print(f"x = {xi:.2f} m: diameter = {di:.3f} m")
    return cross_sec_d

def volume_Sears_Haack_body(R, L, tail_start):
    denominator = (tail_start*(1-tail_start))**(3/4)
    return (R*np.pi/8/denominator)**2 * 3*L/2

def radious_Sears_Haack_body(L, x, volume):
    
    x_factor = (x*(1-x))**(3/4)
    return 8/np.pi * np.sqrt(2*volume/3/L) * x_factor

def R_max_Sears_Haack_body(L, volume):
    return 4/np.pi * np.sqrt(volume/3/L)

def slope_Sears_Haack_body(L, volume, x):
    R = R_max_Sears_Haack_body(L, volume)
    num = 3*R*(1-2*x)
    den = np.sqrt(2)*(x*(1-x))**(1/4)

    if np.any(den == 0):
        den = np.where(den == 0, 1, den)  # avoid division by zero by setting denominator to infinity where it is zero
        num = np.where(den == 0, 0, num)  # if denominator was zero, set slope to zero (horizontal tangent at the tip)
    return num / den
    

def curvature_Sears_Haack_body(L, volume, x):
    R = R_max_Sears_Haack_body(L, volume)
    num = 3*R*(1+4*x*(1-x))
    den = 4*np.sqrt(2)*(x*(1-x))**(5/4) 

    if np.any(den == 0):
        den = np.where(den == 0, 1, den)  # avoid division by zero by setting denominator to infinity where it is zero
        num = np.where(den == 0, 0, num)  # if denominator was zero, set curvature to zero (inflection point at the tip)
    return num / den

def discretize_tail(R, L, tail_start, num_points=10):
    volume = volume_Sears_Haack_body(R, L, tail_start)
    x_tail = np.linspace(L*tail_start, L, num_points)/L
    radius = radious_Sears_Haack_body(L, x_tail, volume)
    slopes = slope_Sears_Haack_body(L, volume, x_tail)
    curvatures = curvature_Sears_Haack_body(L, volume, x_tail)
    return x_tail, radius, slopes, curvatures

def plot_tail(R, L, tail_start, num_points=10, show_plot=True):
    import matplotlib.pyplot as plt

    x_tail, radius, slopes, curvatures = discretize_tail(R, L, tail_start, num_points=num_points)

    _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_tail, radius, 'ro-', ms=6, label=f'Tail profile ({num_points} pts)')

    # tangent line segments at each discretized point
    seg_len = L / (num_points + 2) * 0.03
    for xi, ri, si in zip(x_tail, radius, slopes):
        # tangent unit vector (slope is dr/dx)
        norm = np.sqrt(1 + si**2)
        dx_seg = seg_len / norm
        dr_seg = si * seg_len / norm
        ax.plot([xi - dx_seg, xi + dx_seg],
                [ri - dr_seg, ri + dr_seg],
                color='tomato', lw=1.5, alpha=0.85)
    ax.set_xlabel('x  [m]')
    ax.set_ylabel('Radius  [m]')
    ax.set_title(f'Sears-Haack tail  (R={R} m, L={L:.2f} m, tail starts at {tail_start*100:.1f}% of length)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()




if __name__ == "__main__":
    plot_nose(num_points=15)
    plot_tail(R=R, L=50, tail_start=0.8, num_points=10)
