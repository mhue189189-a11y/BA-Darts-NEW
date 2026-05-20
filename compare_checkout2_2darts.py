# ============================================================
# STRATEGIEVERGLEICH:
# CHECKOUT VON 2 PUNKTEN MIT NUR 2 DARTS
# ============================================================
#
# WICHTIG:
# Es werden exakt dieselben optimalen Radien r2*, r3*
# wie zuvor aus der 3-Dart-Optimierung verwendet.
#
# Der erste Dart r1 entfällt hier vollständig.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0e
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar

# ------------------------------------------------------------
# Board
# ------------------------------------------------------------

R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170

# ------------------------------------------------------------
# Rice PDF
# ------------------------------------------------------------

def rice_pdf(rho, r, sigma):

    x = rho * r / sigma**2

    return (rho / sigma**2) * \
           np.exp(-((rho - r)**2) / (2*sigma**2)) * \
           i0e(x)

# ------------------------------------------------------------
# p_det
# ------------------------------------------------------------

def p_det(r, sigma):

    alpha = np.pi / 20

    def integrand_theta(theta, rho):

        return np.exp(
            -(rho**2 + r**2 - 2*rho*r*np.cos(theta))
            / (2*sigma**2)
        )

    def integrand_rho(rho):

        val, _ = quad(
            integrand_theta,
            -alpha,
            alpha,
            args=(rho,),
            epsabs=1e-8,
            epsrel=1e-8
        )

        return rho * val

    val, _ = quad(
        integrand_rho,
        R_DOUBLE_INNER,
        R_DOUBLE_OUTER,
        epsabs=1e-8,
        epsrel=1e-8
    )

    return val / (2 * np.pi * sigma**2)

# ------------------------------------------------------------
# q_det
# ------------------------------------------------------------

def Q_det(r, sigma):

    val, _ = quad(
        lambda rho: rice_pdf(rho, r, sigma),
        0,
        R_DOUBLE_OUTER,
        epsabs=1e-10,
        epsrel=1e-10
    )

    return val

def q_det(r, sigma):

    return Q_det(r, sigma) - p_det(r, sigma)

# ------------------------------------------------------------
# Optimales r3
# ------------------------------------------------------------

def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded'
    )

    return res.x, p_det(res.x, sigma)

# ------------------------------------------------------------
# 3-DART OPTIMIERUNG
# (liefert dieselben r2*, r3* wie zuvor)
# ------------------------------------------------------------

def checkout_value_det(r, rs, P_det_vals, Q_det_vals, r3):

    r1, r2 = r

    p1 = np.interp(r1, rs, P_det_vals)
    q1 = np.interp(r1, rs, Q_det_vals)

    p2 = np.interp(r2, rs, P_det_vals)
    q2 = np.interp(r2, rs, Q_det_vals)

    p3 = np.interp(r3, rs, P_det_vals)

    val = p1 + \
          (1 - p1 - q1) * \
          (p2 + (1 - p2 - q2) * p3)

    return -val

def optimize_r1_r2_det(rs, P_det_vals, Q_det_vals, r3):

    r_init = rs[np.argmax(P_det_vals)]

    res = minimize(
        checkout_value_det,
        x0=[r_init, r_init],
        args=(rs, P_det_vals, Q_det_vals, r3),
        bounds=[(0, 250), (0, 250)],
        method='L-BFGS-B'
    )

    return res.x[0], res.x[1], -res.fun

def optimal_strategy_3darts(sigma):

    rs = np.linspace(0, 250, 120)

    P_det_vals = np.array([
        p_det(r, sigma)
        for r in rs
    ])

    Q_det_vals = np.array([
        q_det(r, sigma)
        for r in rs
    ])

    r3, _ = find_r3_det(sigma)

    r1_opt, r2_opt, val_opt = optimize_r1_r2_det(
        rs,
        P_det_vals,
        Q_det_vals,
        r3
    )

    return r1_opt, r2_opt, r3

# ------------------------------------------------------------
# 2-DART CHECKOUT
# mit denselben r2*, r3*
# ------------------------------------------------------------

def checkout_2darts(r2, r3, sigma):

    p2 = p_det(r2, sigma)
    q2 = q_det(r2, sigma)

    p3 = p_det(r3, sigma)

    return p2 + (1 - p2 - q2) * p3

# ------------------------------------------------------------
# Vergleich der Strategien
# ------------------------------------------------------------

def compare_2dart_strategies():

    sigmas = np.linspace(1, 80, 50)

    vals_opt = []
    vals_naive = []
    vals_greedy = []

    r2_list = []
    r3_list = []

    for sigma in sigmas:

        print(f"σ = {sigma:.1f}")

        # ----------------------------------------------------
        # OPTIMALE 3-DART STRATEGIE
        # -> dieselben r2*, r3*
        # ----------------------------------------------------

        r1_opt, r2_opt, r3_opt = optimal_strategy_3darts(sigma)

        r2_list.append(r2_opt)
        r3_list.append(r3_opt)

        # ----------------------------------------------------
        # OPTIMAL
        # ----------------------------------------------------

        val_opt = checkout_2darts(
            r2_opt,
            r3_opt,
            sigma
        )

        vals_opt.append(val_opt)

        # ----------------------------------------------------
        # NAIV
        # ----------------------------------------------------

        r_naive = 166

        val_naive = checkout_2darts(
            r_naive,
            r_naive,
            sigma
        )

        vals_naive.append(val_naive)

        # ----------------------------------------------------
        # GREEDY
        # ----------------------------------------------------

        r_greedy, _ = find_r3_det(sigma)

        val_greedy = checkout_2darts(
            r_greedy,
            r_greedy,
            sigma
        )

        vals_greedy.append(val_greedy)

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    vals_opt = np.array(vals_opt)
    vals_naive = np.array(vals_naive)
    vals_greedy = np.array(vals_greedy)

    r2_list = np.array(r2_list)
    r3_list = np.array(r3_list)

    # --------------------------------------------------------
    # PLOT 1:
    # Strategievergleich
    # --------------------------------------------------------

    plt.figure(figsize=(9,6))

    plt.plot(
        sigmas,
        vals_opt,
        linewidth=2,
        label="Optimal"
    )

    plt.plot(
        sigmas,
        vals_naive,
        '--',
        label="Naiv"
    )

    plt.plot(
        sigmas,
        vals_greedy,
        ':',
        label="Greedy"
    )

    plt.plot(
        sigmas,
        vals_opt - vals_greedy,
        label="Opt - Greedy",
        alpha=0.8
    )

    plt.plot(
        sigmas,
        vals_opt - vals_naive,
        label="Opt - Naiv",
        alpha=0.8
    )

    plt.xlabel("σ (mm)")
    plt.ylabel("Checkout-Wahrscheinlichkeit")

    plt.title(
        "Checkout von 2 Punkten mit 2 Darts"
    )

    plt.legend()
    plt.grid()

    plt.show()

    # --------------------------------------------------------
    # PLOT 2:
    # Verwendete Radien
    # --------------------------------------------------------

    plt.figure(figsize=(9,6))

    plt.plot(
        sigmas,
        r2_list,
        linewidth=2,
        label=r"$r_2^\ast$"
    )

    plt.plot(
        sigmas,
        r3_list,
        linewidth=2,
        label=r"$r_3^\ast$"
    )

    plt.xlabel("σ (mm)")
    plt.ylabel("Optimaler Radius (mm)")

    plt.title(
        "Verwendete optimale Radien für 2 Darts"
    )

    plt.legend()
    plt.grid()

    plt.show()

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    compare_2dart_strategies()
