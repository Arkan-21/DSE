import numpy as np


def boundary_layer_edge_properties(
    M_e,
    P_e,
    T_e,
    x,
    T_w,
    gamma=1.4,
    Pr=0.71,
    R=287.05,
):
    """
    Computes H_tr and Cf from the Ding et al.
    hypersonic BLDT transformation method.

    Inputs
    ------
    M_e : edge Mach number
    P_e : edge pressure [Pa]
    T_e : edge temperature [K]
    x   : distance from leading edge [m]
    T_w : wall temperature [K]

    Returns
    -------
    dict containing:
        H_tr
        Cf
        Re_x
        Re_x_star
        Re_x_eq
        T_aw
        T_star
        mu_e
        mu_star
    """

    # --------------------------------------------------
    # Eq. (19)
    # --------------------------------------------------

    lam = Pr**(1/3)

    T_aw = T_e * (1.0+ lam * (gamma - 1.0)/2.0 * M_e**2)

    # --------------------------------------------------
    # Eq. (23)
    # --------------------------------------------------

    T0 = T_e * (1.0 + (gamma - 1.0)/2.0 * M_e**2)

    # --------------------------------------------------
    # Eq. (18)
    # --------------------------------------------------

    T_star = (0.5 * T_w + 0.22 * T_aw+ 0.28 * T_e)

    # --------------------------------------------------
    # Eq. (21)
    # --------------------------------------------------

    mu_star = ( 1.458e-6* T_star**1.5/ (T_star + 110.4))

    mu_e = (1.458e-6* T_e**1.5/ (T_e + 110.4))

    # --------------------------------------------------
    # Eq. (22)
    # --------------------------------------------------

    mu0 = (1.458e-6* T0**1.5/ (T0 + 110.4))

    # --------------------------------------------------
    # Edge properties
    # --------------------------------------------------

    rho_e = P_e / (R * T_e)

    a_e = np.sqrt(gamma * R * T_e)

    u_e = M_e * a_e

    # --------------------------------------------------
    # Eq. (17)
    # --------------------------------------------------

    Re_x = rho_e * u_e * x / mu_e

    # --------------------------------------------------
    # Eq. (16)
    # --------------------------------------------------

    Re_x_star = ((T_e / T_star)* (mu_star / mu_e)* Re_x)

    # --------------------------------------------------
    # Eq. (15)
    # Equivalent incompressible Reynolds number
    # --------------------------------------------------

    A = mu_star * Re_x_star / mu0

    logA = np.log10(A)

    Re_x_eq = (A)/ (logA - 2.3686)* (  ((logA - 1.5)**3/(np.log10(Re_x_star) - 1.5)**2- 0.8686))

    # --------------------------------------------------
    # Eq. (14)
    # --------------------------------------------------

    logRe = np.log10(Re_x_eq)

    Cfi = (0.088* (logRe - 2.3686) / (logRe - 1.5)**3)

    # --------------------------------------------------
    # Eq. (13)
    # --------------------------------------------------

    H_i = 1.0 / (1.0- 7.0*np.sqrt(Cfi/2.0))

    # --------------------------------------------------
    # Eq. (11)
    # --------------------------------------------------

    H_tr = H_i * T_w/T0 + T_aw/T0- 1.0

    # --------------------------------------------------
    # Eq. (12)
    # --------------------------------------------------

    Cf = (T_e/T_star) * Cfi

    return {
        "H_tr": H_tr,
        "Cf": Cf,
        "Re_x": Re_x,
        "Re_x_star": Re_x_star,
        "Re_x_eq": Re_x_eq,
        "H_i": H_i,
        "T_aw": T_aw,
        "T_star": T_star,
        "rho_e": rho_e,
        "u_e": u_e,
        "mu_e": mu_e,
        "mu_star": mu_star,
    }


def rk4_step(f, x, y, dx):
    """
    One RK4 step

    dy/dx = f(x,y)

    Parameters
    ----------
    f : callable
    x : current x
    y : current solution
    dx : step size

    Returns
    -------
    y_next
    """

    k1 = f(x, y)

    k2 = f(x + 0.5*dx,y + 0.5*dx*k1)

    k3 = f(x + 0.5*dx,y + 0.5*dx*k2)

    k4 = f(x + dx,y + dx*k3)

    return y + dx*(k1 + 2*k2 + 2*k3 + k4)/6


def rk4_integrate(f, x0, xf, y0, n_steps):

    x = np.linspace(x0, xf, n_steps + 1)

    y = np.zeros_like(x)

    y[0] = y0

    dx = (xf - x0)/n_steps

    for i in range(n_steps):

        y[i+1] = rk4_step(f,x[i],y[i],dx)

    return x, y

def dphi_dx(
    x,
    phi_tr,
    M_e,
    dMdx,
    r,
    drdx,
    H_tr,
    Cf,
    T_e,
    T0,
    theta,
    gamma=1.4
):

    term1 = -(phi_tr/M_e* (2 + H_tr)* dMdx)

    #term2 = (-phi_tr/r* drdx)

    term3 = (0.5*Cf* (1/np.cos(theta))* (T_e/T0)**((gamma+1)/(2*(gamma-1))))

    return term1 + term3

if __name__ == "__main__":
    """
    Example usage of the boundary layer edge properties function.
    """
    # --------------------------------------------------
    # Example edge conditions
    # (replace with intake solver outputs)
    # --------------------------------------------------
    gamma = 1.4
    M_e = 3.55
    P_e = 6637.5     # Pa
    T_e = 797.92        # K
    T_w = 600.0        # K
    x = 4            # m

    phi0 = 0.000
    theta_wall = np.radians(9.8)      # flat wall
    r = 1                  # planar flow
    drdx = 0.02
    dMdx = 0.2

    # --------------------------------------------------
    # Compute H_tr and Cf
    # --------------------------------------------------

    
    bl = boundary_layer_edge_properties(
        M_e=M_e,
        P_e=P_e,
        T_e=T_e,
        x=x,
        T_w=T_w,
    )

    H_tr = bl["H_tr"]
    Cf   = bl["Cf"]
    Re_x = bl["Re_x"]

    print("\nBoundary Layer Properties")
    print("-" * 40)
    print(f"H_tr       = {H_tr:.6f}")
    print(f"Cf         = {Cf:.6e}")
    print(f"Re_x       = {bl['Re_x']:.6e}")
    print(f"T_aw       = {bl['T_aw']:.2f} K")
    print(f"T_star     = {bl['T_star']:.2f} K")

      # --------------------------------------------------
    # Example RK4 integration
    # --------------------------------------------------

    T0 = T_e * (1.0 +(gamma - 1.0)/2.0 * M_e**2)

    def phi_rhs(x, phi_tr):

        bl = boundary_layer_edge_properties(
        M_e=M_e,
        P_e=P_e,
        T_e=T_e,
        x=max(x,1e-6),
        T_w=T_w,)

        return dphi_dx(
        x=x,
        phi_tr=phi_tr,
        M_e=M_e,
        dMdx=dMdx,
        r=r,
        drdx=drdx,
        H_tr=bl["H_tr"],
        Cf=bl["Cf"],
        T_e=T_e,
        T0=T0,
        theta=theta_wall)

   

    x_sol, phi_sol = rk4_integrate(
        f=phi_rhs,
        x0=0.0,
        xf=x,
        y0=phi0,
        n_steps=500
    )

    phi = phi_sol*(T0/T_e)**((gamma+1)/(2*(gamma-1)))
    H = H_tr*(T0/T_e)+(T0/T_e)-1

    delta_star = phi * H

    print("\nRK4 Integration")
    print("-" * 40)
    print(f"phi_tr(0)   = {phi_sol[0]:.6e}")
    print(f"phi_tr(end) = {phi_sol[-1]:.6e}")

    # --------------------------------------------------
    # Plot result
    # --------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))

    plt.plot(
        x_sol,
        delta_star,
        lw=2, label= "Numerical method of momentum integration"
    )

    delta_star2 = x_sol * 0.046* Re_x**(-0.2) +phi0

    plt.plot(x_sol , delta_star2, lw=2, ls="--", label="Power law method")

    plt.xlabel("x [m]")
    plt.ylabel(r"$\delta^*$ [m]")
    plt.title("Boundary Layer Displacement Thickness (BLDT)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()