# ============================================================
# CHECKOUT VON 3 PUNKTEN MIT 2 DARTS
# VOLLSTÄNDIG DETERMINISTISCH
#
# WICHTIGE ÄNDERUNG:
#
# Für σ >= 11 wird r1_opt konstant auf dem
# Grenzwert 137.12 mm gehalten,
# da p1(r) dort praktisch flach wird.
#
# Dadurch:
# - stabilere Strategie
# - einfachere Interpretation
# - praktisch identische Ergebnisse
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import i0

# ============================================================
# BOARD
# ============================================================

R_BULL_INNER = 6.35
R_BULL_OUTER = 15.9

R_TRIPLE_INNER = 99
R_TRIPLE_OUTER = 107

R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170

segments = np.array([
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
])

ANGLE_PER_SEG = 2*np.pi/20

ALPHA = np.pi/20

# ============================================================
# FIXIERTER GRENZWERT
# ============================================================

R1_LIMIT = 137.12

# ============================================================
# DICHTE
# ============================================================

def gaussian_density(x, y, mu_x, mu_y, sigma):

    return (
        1/(2*np.pi*sigma**2)
        * np.exp(
            -((x-mu_x)**2 + (y-mu_y)**2)
            /(2*sigma**2)
        )
    )

# ============================================================
# p_det
# ============================================================

def p_det(r, sigma):

    def integrand(theta, rho):

        x = rho*np.cos(theta)
        y = rho*np.sin(theta)

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
        R_DOUBLE_OUTER
    )

    return val

# ============================================================
# p1_det
# ============================================================

def p1_det(r, sigma):

    radial_intervals = [
        (R_BULL_OUTER, R_TRIPLE_INNER),
        (R_TRIPLE_OUTER, R_DOUBLE_INNER)
    ]

    total = 0

    for r_min, r_max in radial_intervals:

        def integrand(theta, rho):

            x = rho*np.cos(theta)
            y = rho*np.sin(theta)

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
            r_max
        )

        total += val

    return total

# ============================================================
# Q_det
# ============================================================

def rice_pdf(rho, r, sigma):

    return (
        rho/sigma**2
        * np.exp(-(rho**2 + r**2)/(2*sigma**2))
        * i0(rho*r/sigma**2)
    )

def Q_det(r, sigma):

    val, _ = quad(
        lambda rho:
            rice_pdf(rho, r, sigma),
        0,
        R_DOUBLE_OUTER
    )

    return val

def q_det(r, sigma):

    return Q_det(r, sigma) - p_det(r, sigma)

# ============================================================
# r3 optimal
# ============================================================

def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded'
    )

    return res.x

# ============================================================
# r1 optimal (INTERVALLWEISE DEFINIERT)
# ============================================================

def find_r1_det(sigma):

    # --------------------------------------------------------
    # Für große σ:
    # flaches Maximum -> konstant
    # --------------------------------------------------------

    if sigma <= 10:

        return R1_LIMIT

    # --------------------------------------------------------
    # Sonst echte Optimierung
    # --------------------------------------------------------

    res = minimize_scalar(
        lambda r: -p1_det(r, sigma),
        bounds=(20, 170),
        method='bounded'
    )

    return res.x

# ============================================================
# CHECKOUT 3 MIT 2 DARTS
# ============================================================

def checkout_3_two_darts(r1, r2, sigma):

    return (
        p1_det(r1, sigma)
        * p_det(r2, sigma)
    )

# ============================================================
# PLOT p1(r)
# ============================================================

def plot_p1_det(sigma=1):

    rs = np.linspace(0, 200, 400)

    vals = np.array([
        p1_det(r, sigma)
        for r in rs
    ])

    r_opt = find_r1_det(sigma)

    plt.figure(figsize=(8,5))

    plt.plot(
        rs,
        vals,
        linewidth=2,
        label=r"$p_1(r)$"
    )

    plt.axvline(
        r_opt,
        linestyle='--',
        linewidth=2,
        label=rf"$r_1^*={r_opt:.2f}$ mm"
    )

    if sigma <= 10:

        plt.axvline(
            R1_LIMIT,
            linestyle=':',
            linewidth=2,
            label=rf"Fixierter Grenzwert {R1_LIMIT:.2f} mm"
        )

    plt.xlabel("Radius r (mm)")
    plt.ylabel(r"$p_1(r)$")

    plt.title(
        rf"Deterministische Wahrscheinlichkeit $p_1(r)$ ($\sigma={sigma}$)"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# STRATEGIEVERGLEICH
# ============================================================

def compare_checkout_3_two_darts():

    # --------------------------------------------------------
    # Jetzt wieder ab σ = 1
    # --------------------------------------------------------

    sigmas = np.linspace(1.9,55, 75)

    vals_opt = []
    vals_naive = []

    r1_list = []
    r2_list = []

    # --------------------------------------------------------
    # Naive Strategie
    # --------------------------------------------------------

    r1_naive = (R_TRIPLE_OUTER + R_DOUBLE_INNER)/2
    r2_naive = 166

    for sigma in sigmas:

        print(f"σ = {sigma:.1f}")

        # ----------------------------------------------------
        # Optimal / Greedy
        # ----------------------------------------------------

        r1_opt = find_r1_det(sigma)
        r2_opt = find_r3_det(sigma)

        val_opt = checkout_3_two_darts(
            r1_opt,
            r2_opt,
            sigma
        )

        vals_opt.append(val_opt)

        r1_list.append(r1_opt)
        r2_list.append(r2_opt)

        # ----------------------------------------------------
        # Naiv
        # ----------------------------------------------------

        val_naive = checkout_3_two_darts(
            r1_naive,
            r2_naive,
            sigma
        )

        vals_naive.append(val_naive)

    vals_opt = np.array(vals_opt)
    vals_naive = np.array(vals_naive)

    # ========================================================
    # PLOT 1
    # ========================================================

    plt.figure(figsize=(9,6))

    plt.plot(
        sigmas,
        vals_opt,
        linewidth=2,
        label="Optimal / Greedy"
    )

    plt.plot(
        sigmas,
        vals_naive,
        '--',
        linewidth=2,
        label="Naiv"
    )

    plt.plot(
        sigmas,
        vals_opt - vals_naive,
        linewidth=2,
        label="Differenz"
    )

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Checkout-Wahrscheinlichkeit")

    plt.title(
        "3 Punkte mit 2 Darts"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ========================================================
    # PLOT 2
    # ========================================================

    plt.figure(figsize=(9,6))

    plt.plot(
        sigmas,
        r1_list,
        linewidth=2,
        label=r"Optimal: $r_2^*$"
    )

    plt.plot(
        sigmas,
        r2_list,
        linewidth=2,
        label=r"Optimal/Greedy: $r_3^*$"
    )

    # naive Radien

    plt.axhline(
        r1_naive,
        linestyle='--',
        linewidth=2,
        label=rf"Naiv: $r_2={r1_naive:.1f}$ mm"
    )

    plt.axhline(
        r2_naive,
        linestyle=':',
        linewidth=2,
        label=rf"Naiv: $r_3={r2_naive:.1f}$ mm"
    )

    # Grenzwert markieren

    plt.axhline(
        R1_LIMIT,
        linestyle='-.',
        linewidth=2,
        label=rf"Grenzwert $r_2={R1_LIMIT:.2f}$ mm"
    )

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Optimaler Radius (mm)")

    plt.title(
        "Optimale und naive Zielradien"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

   # ========================================================
    # DIFFERENZPLOT
    # ========================================================

    plt.figure(figsize=(9,6))

    plt.plot(
        sigmas,
        vals_opt - vals_naive,
        linewidth=2,
        label="Absolute Differenz"
    )

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Differenz")

    plt.title(
        "Differenz zwischen optimaler und naiver Strategie"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    sigma = 5

    # --------------------------------------------------------
    # Deterministischer p1 Plot
    # --------------------------------------------------------

    plot_p1_det(sigma=sigma)

    # --------------------------------------------------------
    # Strategievergleich
    # --------------------------------------------------------

    compare_checkout_3_two_darts()
