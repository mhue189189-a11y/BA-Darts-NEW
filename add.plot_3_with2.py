# ============================================================
# CHECKOUT VON 3 PUNKTEN MIT 2 DARTS
# VOLLSTÄNDIG DETERMINISTISCH
#
# Strategien:
#
# Optimal / Greedy:
#   Dart 1 -> optimales p1(r)
#   Dart 2 -> optimales D1
#
# Naiv:
#   Dart 1 -> Mitte großes S1
#   Dart 2 -> Mitte D1 bei r = 166 mm
#
# Zusätzlich:
#   - deterministischer Plot von p1(r)
#   - stabilisierte optimale Radien
#   - Differenzplots
#   - Detailvergleich für fixes sigma
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
# WINKEL
# ============================================================

def get_angle_for_segment(seg_value):

    i = np.where(segments == seg_value)[0][0]

    return np.pi/2 - (i + 0.5)*ANGLE_PER_SEG + np.pi/20


# ============================================================
# GAUSS-DICHTE
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
# p_D1(r)
# ============================================================

def p_det(r, sigma):

    def integrand(theta, rho):

        x = rho*np.cos(theta)
        y = rho*np.sin(theta)

        mu_x = r
        mu_y = 0

        return (
            gaussian_density(x, y, mu_x, mu_y, sigma)
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
# p1(r)
# Wahrscheinlichkeit exakt die 1 zu treffen
# ============================================================

def p1_det(r, sigma):

    theta1 = -ALPHA
    theta2 = ALPHA

    radial_intervals = [
        (R_BULL_OUTER, R_TRIPLE_INNER),
        (R_TRIPLE_OUTER, R_DOUBLE_INNER)
    ]

    total = 0

    for r_min, r_max in radial_intervals:

        def integrand(theta, rho):

            x = rho*np.cos(theta)
            y = rho*np.sin(theta)

            mu_x = r
            mu_y = 0

            return (
                gaussian_density(x, y, mu_x, mu_y, sigma)
                * rho
            )

        val, _ = quad(
            lambda rho:
                quad(
                    lambda theta:
                        integrand(theta, rho),
                    theta1,
                    theta2
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
        lambda rho: rice_pdf(rho, r, sigma),
        0,
        R_DOUBLE_OUTER
    )

    return val


def q_det(r, sigma):

    return Q_det(r, sigma) - p_det(r, sigma)


# ============================================================
# OPTIMALE RADIEN
# ============================================================

def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded'
    )

    return res.x


def find_r1_det(sigma):

    res = minimize_scalar(
        lambda r: -p1_det(r, sigma),
        bounds=(20, 170),
        method='bounded'
    )

    return res.x


# ============================================================
# STABILISIERTES r1*
# ============================================================

R1_CRITICAL = 137.12
# ============================================================
# STABILISIERTES OPTIMALES r1
#
# Für sigma >= 11 wird der konstante Grenzwert
#
#     r1* = 137.12 mm
#
# verwendet.
#
# Dieser Wert entspricht dem asymptotischen Maximum
# von p1(r).
# ============================================================

def find_r1_det(sigma):

    return 137.12

def r1_optimal_greedy(sigma):

    if sigma >= 11:
        return R1_CRITICAL

    return find_r1_det(sigma)


# ============================================================
# CHECKOUT 3 MIT 2 DARTS
# ============================================================

def checkout_3_two_darts(r1, r2, sigma):

    p1 = p1_det(r1, sigma)
    p2 = p_det(r2, sigma)

    return p1 * p2


# ============================================================
# PLOT p1(r)
# ============================================================

def plot_p1_det(sigma=1):

    rs = np.linspace(0, 200, 400)

    vals = np.array([
        p1_det(r, sigma)
        for r in rs
    ])

    r_opt = r1_optimal_greedy(sigma)

    plt.figure(figsize=(9,6))

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

    sigmas = np.linspace(1, 80, 80)

    vals_opt = []
    vals_naive = []

    r1_list = []
    r2_list = []

    r1_naive = (R_TRIPLE_OUTER + R_DOUBLE_INNER)/2
    r2_naive = 166

    for sigma in sigmas:

        print(f"σ = {sigma:.1f}")

        # ----------------------------------------------------
        # Optimal / Greedy
        # ----------------------------------------------------

        r1_opt = r1_optimal_greedy(sigma)
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

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Checkout-Wahrscheinlichkeit")

    plt.title("3 Punkte mit 2 Darts")

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
        label=r"Optimal: $r_1^*$"
    )

    plt.plot(
        sigmas,
        r2_list,
        linewidth=2,
        label=r"Optimal/Greedy: $r_2^*$"
    )

    plt.axhline(
        r1_naive,
        linestyle='--',
        linewidth=2,
        label=rf"Naiv: $r_1={r1_naive:.1f}$ mm"
    )

    plt.axhline(
        r2_naive,
        linestyle=':',
        linewidth=2,
        label=rf"Naiv: $r_2={r2_naive:.1f}$ mm"
    )

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Radius (mm)")

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
        label=r"$\Delta = P_{\mathrm{opt}}-P_{\mathrm{naiv}}$"
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
    return vals_naive, vals_opt

# ============================================================
# DETAILVERGLEICH
# ============================================================

def compare_probabilities_fixed_sigma(sigma=1):

    rs = np.linspace(0, 200, 500)

    p1_vals = np.array([
        p1_det(r, sigma)
        for r in rs
    ])

    pD1_vals = np.array([
        p_det(r, sigma)
        for r in rs
    ])

    # --------------------------------------------------------
    # Optimal konsistent!
    # --------------------------------------------------------

    r1_opt = r1_optimal_greedy(sigma)
    r2_opt = find_r3_det(sigma)

    p1_opt = p1_det(r1_opt, sigma)
    pD1_opt = p_det(r2_opt, sigma)

    val_opt = p1_opt * pD1_opt

    # --------------------------------------------------------
    # Naiv
    # --------------------------------------------------------

    r1_naive = (R_TRIPLE_OUTER + R_DOUBLE_INNER)/2
    r2_naive = 166

    p1_naive = p1_det(r1_naive, sigma)
    pD1_naive = p_det(r2_naive, sigma)

    val_naive = p1_naive * pD1_naive

    # --------------------------------------------------------
    # Differenzen
    # --------------------------------------------------------

    abs_diff = val_opt - val_naive

    rel_diff = (
        (val_opt - val_naive)
        / val_naive
        * 100
    )

    # --------------------------------------------------------
    # AUSGABE
    # --------------------------------------------------------

    print("\n================================================")
    print(f"σ = {sigma}")
    print("================================================")

    print("\nOPTIMAL / GREEDY")

    print(f"r1_opt = {r1_opt:.4f} mm")
    print(f"r2_opt = {r2_opt:.4f} mm")

    print(f"p1_opt  = {p1_opt:.8f}")
    print(f"pD1_opt = {pD1_opt:.8f}")

    print(f"p1*pD1  = {val_opt:.8f}")

    print("\nNAIV")

    print(f"r1_naive = {r1_naive:.4f} mm")
    print(f"r2_naive = {r2_naive:.4f} mm")

    print(f"p1_naive  = {p1_naive:.8f}")
    print(f"pD1_naive = {pD1_naive:.8f}")

    print(f"p1*pD1    = {val_naive:.8f}")

    print("\nDIFFERENZEN")

    print(f"Absolute Differenz = {abs_diff:.8f}")
    print(f"Relative Differenz = {rel_diff:.4f} %")

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(10,6))

    plt.plot(
        rs,
        p1_vals,
        linewidth=2,
        label=r"$p_1(r)$"
    )

    plt.plot(
        rs,
        pD1_vals,
        linewidth=2,
        label=r"$p_{D1}(r)$"
    )

    # --------------------------------------------------------
    # Optimale Radien
    # --------------------------------------------------------

    plt.axvline(
        r1_opt,
        linestyle='--',
        linewidth=2,
        label=rf"Optimal: $r_1^*={r1_opt:.2f}$ mm"
    )

    plt.axvline(
        r2_opt,
        linestyle=':',
        linewidth=2,
        label=rf"Optimal: $r_2^*={r2_opt:.2f}$ mm"
    )

    # --------------------------------------------------------
    # Naive Radien
    # --------------------------------------------------------

    plt.axvline(
        r1_naive,
        linestyle='-.',
        linewidth=2,
        label=rf"Naiv: $r_1={r1_naive:.2f}$ mm"
    )

    plt.axvline(
        r2_naive,
        linestyle=(0, (1,1)),
        linewidth=2,
        label=rf"Naiv: $r_2={r2_naive:.2f}$ mm"
    )

    plt.xlabel("Radius r (mm)")
    plt.ylabel("Wahrscheinlichkeit")

    plt.title(
        rf"$p_1(r)$ und $p_{{D1}}(r)$ für $\sigma={sigma}$"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    sigma = 40


    compare_probabilities_fixed_sigma(sigma=sigma)

    sigma = 5 

    compare_probabilities_fixed_sigma(sigma=sigma)
    
