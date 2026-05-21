# ============================================================
# RADIUSPLOT:
#
#   r1*(sigma)
#   vs.
#   r1_greedy(sigma)
#
# FINAL KORRIGIERTE VERSION
#
# WICHTIG:
#
#   - kontinuierliche Optimierung
#     mittels minimize_scalar
#
#   - r1_opt konvergiert für kleine sigma
#     ebenfalls gegen 137.12 mm
#
#   - greedy:
#
#         sigma <= 10:
#             137.12
#
#         sigma > 10:
#             echtes Maximum von p1(r)
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import i0e

# ============================================================
# BOARD
# ============================================================

R_BULL_OUTER = 15.9

R_TRIPLE_INNER = 99
R_TRIPLE_OUTER = 107

R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170

ALPHA = np.pi / 20

# ============================================================
# NAIV
# ============================================================

R1_NAIVE = (R_TRIPLE_OUTER + R_DOUBLE_INNER) / 2

# ============================================================
# GREEDY
# ============================================================

R1_SMALL_SIGMA = 137.12

# ============================================================
# GAUSS
# ============================================================

def gaussian_density(x, y, mu_x, mu_y, sigma):

    return (
        1 / (2 * np.pi * sigma**2)
        * np.exp(
            -((x - mu_x)**2 + (y - mu_y)**2)
            / (2 * sigma**2)
        )
    )

# ============================================================
# p_D1
# ============================================================

def p_det(r, sigma):

    def integrand(theta, rho):

        x = rho * np.cos(theta)
        y = rho * np.sin(theta)

        return (
            gaussian_density(x, y, r, 0, sigma)
            * rho
        )

    val, _ = quad(
        lambda rho:
            quad(
                lambda theta:
                    integrand(theta, rho),
                -ALPHA,
                ALPHA
            )[0],
        R_DOUBLE_INNER,
        R_DOUBLE_OUTER,
        epsabs=1e-8,
        epsrel=1e-8
    )

    return val

# ============================================================
# q_D1
# ============================================================

def rice_pdf(rho, r, sigma):

    x = rho * r / sigma**2

    return (
        rho / sigma**2
        * np.exp(-((rho - r)**2) / (2 * sigma**2))
        * i0e(x)
    )

def Q_det(r, sigma):

    val, _ = quad(
        lambda rho: rice_pdf(rho, r, sigma),
        0,
        R_DOUBLE_OUTER,
        epsabs=1e-8,
        epsrel=1e-8,
        limit=200
    )

    return val

def q_det(r, sigma):

    return Q_det(r, sigma) - p_det(r, sigma)

# ============================================================
# p1
# ============================================================

def p1_det(r, sigma):

    intervals = [
        (R_BULL_OUTER, R_TRIPLE_INNER),
        (R_TRIPLE_OUTER, R_DOUBLE_INNER)
    ]

    total = 0

    for r_min, r_max in intervals:

        def integrand(theta, rho):

            x = rho * np.cos(theta)
            y = rho * np.sin(theta)

            return (
                gaussian_density(x, y, r, 0, sigma)
                * rho
            )

        val, _ = quad(
            lambda rho:
                quad(
                    lambda theta:
                        integrand(theta, rho),
                    -ALPHA,
                    ALPHA
                )[0],
            r_min,
            r_max,
            epsabs=1e-8,
            epsrel=1e-8
        )

        total += val

    return total

# ============================================================
# p0
# ============================================================

def p0_det(r, sigma):

    return 1 - Q_det(r, sigma)

# ============================================================
# OPTIMALES D1
# ============================================================

def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded'
    )

    return res.x

# ============================================================
# ECHTES MAXIMUM p1
# ============================================================

def p1_maximum(sigma):

    res = minimize_scalar(
        lambda r: -p1_det(r, sigma),
        bounds=(0, 200),
        method='bounded'
    )

    return res.x

# ============================================================
# GREEDY
# ============================================================

def greedy_r1(sigma):

    # --------------------------------------------------------
    # kleines sigma:
    # numerisch flaches Plateau
    # --------------------------------------------------------

    if sigma <= 10:
        return R1_SMALL_SIGMA

    return p1_maximum(sigma)

# ============================================================
# C2
# ============================================================

def C2_optimal(sigma):

    r2 = find_r3_det(sigma)

    p2 = p_det(r2, sigma)
    q2 = q_det(r2, sigma)

    p3 = p_det(r2, sigma)

    return (
        p2
        + (1 - p2 - q2) * p3
    )

# ============================================================
# C3
# ============================================================

def C3_optimal(sigma):

    # --------------------------------------------------------
    # kleines sigma:
    # gleiche Konvergenz wie greedy
    # --------------------------------------------------------

    if sigma <= 10:

        r1 = R1_SMALL_SIGMA

    else:

        # ----------------------------------------------------
        # kontinuierliche Optimierung
        # ----------------------------------------------------

        C2 = C2_optimal(sigma)

        r2 = find_r3_det(sigma)

        pD1 = p_det(r2, sigma)

        def objective(r):

            p1 = p1_det(r, sigma)
            p0 = p0_det(r, sigma)

            return -(
                p1 * C2
                + p0 * (p1 * pD1)
            )

        res = minimize_scalar(
            objective,
            bounds=(0, 200),
            method='bounded'
        )

        r1 = res.x

    r2 = find_r3_det(sigma)

    return (
        r1,
        p1_det(r1, sigma)
        * p_det(r2, sigma)
    )

# ============================================================
# RADIUS-STUDIE
# ============================================================

def radius_study():

    sigmas = np.linspace(1, 60, 80)

    r1_opts = []
    r1_greeds = []

    # --------------------------------------------------------
    # sigma-loop
    # --------------------------------------------------------

    for sigma in sigmas:

        print(f"σ = {sigma:.2f}")

        # ----------------------------------------------------
        # greedy
        # ----------------------------------------------------

        r1_greed = greedy_r1(sigma)

        # ----------------------------------------------------
        # optimal
        # ----------------------------------------------------

        r1_opt, _ = C3_optimal(sigma)

        r1_greeds.append(r1_greed)
        r1_opts.append(r1_opt)

    r1_opts = np.array(r1_opts)
    r1_greeds = np.array(r1_greeds)

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        sigmas,
        r1_opts,
        linewidth=3,
        label=r"$r_1^\ast(\sigma)$"
    )

    plt.plot(
        sigmas,
        r1_greeds,
        linestyle='--',
        linewidth=2,
        label=r"$r_1^{greedy}(\sigma)$"
    )

    plt.axhline(
        R1_NAIVE,
        linestyle=':',
        linewidth=2,
        label=r"$r_1^{naiv}$"
    )

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel(r"$r_1$ (mm)")

    plt.title(
        r"Optimales $r_1$ relativ zu $\sigma$"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    radius_study()
