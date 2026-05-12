from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ============================================================
# 1. IDEAL GAS MODEL (replace later with your AirProperties)
# ============================================================
class IdealGas:
    def __init__(self, gamma=1.4, R=287.0):
        self.gamma = gamma
        self.R = R

    def pressure(self, rho, u, E):
        return (self.gamma - 1.0) * (E - 0.5 * rho * u**2)

    def sound_speed(self, p, rho):
        return np.sqrt(self.gamma * p / rho)


# ============================================================
# 2. CONSERVATIVE FLUX-FORM SOLVER
# ============================================================
class FluxForm1D:
    """
    Quasi-1D Euler solver with source terms:

    U = [rho, rho*u, E]

    Handles:
    - area change
    - heat addition
    - mass addition
    - friction (optional extension)
    """

    def __init__(self, gas: IdealGas):
        self.gas = gas

    # -----------------------------
    # Flux vector
    # -----------------------------
    def flux(self, U):
        rho = U[0]
        u = U[1] / rho
        E = U[2]

        p = self.gas.pressure(rho, u, E)

        return np.array([
            rho * u,
            rho * u**2 + p,
            (E + p) * u
        ])

    # -----------------------------
    # RHS of ODE system
    # -----------------------------
    def rhs(self, x, U, geometry_fn, source_fn):

        rho, u, E = U
        p = self.gas.pressure(rho, u, E)

        A, dA_dx = geometry_fn(x)
        sources = source_fn(x)

        mdot_src = sources.get("mdot", 0.0)
        q_src = sources.get("qdot", 0.0)

        F = self.flux(U)

        dUdx = np.zeros(3)

        # -----------------------------
        # MASS
        # -----------------------------
        dUdx[0] = mdot_src

        # -----------------------------
        # MOMENTUM
        # -----------------------------
        dUdx[1] = (
            -F[1] / A * dA_dx
            + p * dA_dx
        )

        # -----------------------------
        # ENERGY
        # -----------------------------
        dUdx[2] = (
            -F[2] / A * dA_dx
            + q_src
        )

        return dUdx

    # -----------------------------
    # Integrator
    # -----------------------------
    def integrate(self, x0, x1, U0, geometry_fn, source_fn):

        def ode(x, U):
            return self.rhs(x, U, geometry_fn, source_fn)

        sol = solve_ivp(
            ode,
            (x0, x1),
            U0,
            method="BDF",
            rtol=1e-6,
            atol=1e-8,
            max_step=(x1 - x0) / 100
        )

        return sol


# ============================================================
# 3. ENGINE WRAPPER (minimal but functional)
# ============================================================
class SimpleEngine:

    def __init__(self):
        self.gas = IdealGas()
        self.solver = FluxForm1D(self.gas)

    # -----------------------------
    # primitive → conservative
    # -----------------------------
    def to_conservative(self, rho, u, p):
        gamma = self.gas.gamma
        E = p / (gamma - 1.0) + 0.5 * rho * u**2
        return np.array([rho, rho * u, E])

    # -----------------------------
    # initial state
    # -----------------------------
    def inlet(self):
        rho = 1.0
        u = 300*5
        p = 1e5
        return self.to_conservative(rho, u, p)

    # -----------------------------
    # geometry (example nozzle)
    # -----------------------------
    def geometry(self, x):
        A0 = 1.0
        A1 = 2.0

        A = A0 + (A1 - A0) * x
        dA_dx = (A1 - A0)

        return A, dA_dx

    # -----------------------------
    # source terms (heat + fuel)
    # -----------------------------
    def source(self, x):

        qdot = 2e5 if 0.3 < x < 0.6 else 0.0
        mdot = 0.0

        return {
            "qdot": qdot,
            "mdot": mdot
        }

    # -----------------------------
    # run simulation
    # -----------------------------
    def run(self):

        U0 = self.inlet()

        sol = self.solver.integrate(
            x0=0.0,
            x1=1.0,
            U0=U0,
            geometry_fn=self.geometry,
            source_fn=self.source
        )

        x = sol.t
        rho = sol.y[0]
        u = sol.y[1] / rho
        E = sol.y[2]

        p = np.array([
            self.gas.pressure(rho[i], u[i], E[i])
            for i in range(len(x))
        ])

        return x, rho, u, p


# ============================================================
# 4. DEMO RUN
# ============================================================
if __name__ == "__main__":

    eng = SimpleEngine()
    x, rho, u, p = eng.run()

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(x, rho)
    axs[0].set_ylabel("Density")

    axs[1].plot(x, u)
    axs[1].set_ylabel("Velocity")

    axs[2].plot(x, p)
    axs[2].set_ylabel("Pressure")
    axs[2].set_xlabel("x")

    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.show()