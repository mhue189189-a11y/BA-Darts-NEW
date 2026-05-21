# ============================================================
# VERGLEICHPLOT FÜR FESTES SIGMA
#
# FINAL KORRIGIERTE VERSION
#
# WICHTIG:
#
#   - C2 stammt aus echter 3-Dart-Optimierung
#
#   - C3 stammt aus:
#
#         p1(r1_opt)*pD1(r2_opt)
#
#   - GREEDY:
#
#         sigma <= 10:
#             r = 137.12
#
#         sigma > 10:
#             echtes Maximum von p1(r)
#
#   - p1(r) wird zusätzlich geplottet
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
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
# NAIVE
# ============================================================

R1_NAIVE = (R_TRIPLE_OUTER + R_DOUBLE_INNER) / 2

R2_NAIVE = 166
R3_NAIVE = 166

# ============================================================
# GREEDY
# ============================================================

R1_GREEDY_SMALL_SIGMA = 137.12

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
# OPTIMALE RADII
# ============================================================

def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded'
    )

    return res.x

# ============================================================
# ECHTES MAXIMUM VON p1
# ============================================================

def true_p1_maximum(sigma):

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

    if sigma <= 10:
        return R1_GREEDY_SMALL_SIGMA

    return true_p1_maximum(sigma)

# ============================================================
# 3-DART OPTIMIERUNG
# ============================================================

def checkout_value_det(r, rs, P_det_vals, Q_det_vals, r3):

    r1, r2 = r

    p1 = np.interp(r1, rs, P_det_vals)
    q1 = np.interp(r1, rs, Q_det_vals)

    p2 = np.interp(r2, rs, P_det_vals)
    q2 = np.interp(r2, rs, Q_det_vals)

    p3 = np.interp(r3, rs, P_det_vals)

    val = (
        p1
        + (1 - p1 - q1)
        * (
            p2
            + (1 - p2 - q2) * p3
        )
    )

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

    r3 = find_r3_det(sigma)

    r1_opt, r2_opt, val_opt = optimize_r1_r2_det(
        rs,
        P_det_vals,
        Q_det_vals,
        r3
    )

    return r1_opt, r2_opt, r3, val_opt

# ============================================================
# C2
# ============================================================

def C2_optimal(sigma):

    _, r2_opt, r3_opt, _ = optimal_strategy_3darts(sigma)

    p2 = p_det(r2_opt, sigma)
    q2 = q_det(r2_opt, sigma)

    p3 = p_det(r3_opt, sigma)

    return p2 + (1 - p2 - q2) * p3

def C2_naive(sigma):

    p2 = p_det(R2_NAIVE, sigma)
    q2 = q_det(R2_NAIVE, sigma)

    p3 = p_det(R3_NAIVE, sigma)

    return p2 + (1 - p2 - q2) * p3

# ============================================================
# C3
# ============================================================

def C3_optimal(sigma):

    r1_opt = true_p1_maximum(sigma)
    r2_opt = find_r3_det(sigma)

    return (
        p1_det(r1_opt, sigma)
        * p_det(r2_opt, sigma)
    )

def C3_naive(sigma):

    return (
        p1_det(R1_NAIVE, sigma)
        * p_det(R2_NAIVE, sigma)
    )

def C3_greedy(sigma):

    r1_greed = greedy_r1(sigma)

    r2_opt = find_r3_det(sigma)

    return (
        p1_det(r1_greed, sigma)
        * p_det(r2_opt, sigma)
    )

# ============================================================
# GESAMTWERT
# ============================================================

def total_value(r, sigma, C2, C3):

    p1 = p1_det(r, sigma)
    p0 = p0_det(r, sigma)

    return p1 * C2 + p0 * C3

# ============================================================
# FIXES SIGMA
# ============================================================

def plot_total_value_fixed_sigma(sigma=40):

    rs = np.linspace(0, 200, 800)

    # --------------------------------------------------------
    # Basisfunktionen
    # --------------------------------------------------------

    p1_vals = np.array([
        p1_det(r, sigma)
        for r in rs
    ])

    p0_vals = np.array([
        p0_det(r, sigma)
        for r in rs
    ])

    # --------------------------------------------------------
    # Konstanten
    # --------------------------------------------------------

    C2_opt = C2_optimal(sigma)
    C2_naiv = C2_naive(sigma)

    C3_opt = C3_optimal(sigma)
    C3_naiv = C3_naive(sigma)
    C3_greed = C3_greedy(sigma)

    # --------------------------------------------------------
    # Gesamtwertfunktionen
    # --------------------------------------------------------

    total_opt_vals = np.array([
        total_value(r, sigma, C2_opt, C3_opt)
        for r in rs
    ])

    total_greedy_vals = np.array([
        total_value(r, sigma, C2_opt, C3_greed)
        for r in rs
    ])

    total_naiv_vals = np.array([
        total_value(r, sigma, C2_naiv, C3_naiv)
        for r in rs
    ])

    # --------------------------------------------------------
    # Radien
    # --------------------------------------------------------

    r1_opt = rs[np.argmax(total_opt_vals)]

    r1_greedy = greedy_r1(sigma)

    r1_naive = R1_NAIVE

    # --------------------------------------------------------
    # Werte
    # --------------------------------------------------------

    V_opt = np.max(total_opt_vals)

    V_greedy = total_value(
        r1_greedy,
        sigma,
        C2_opt,
        C3_greed
    )

    V_naiv = total_value(
        r1_naive,
        sigma,
        C2_naiv,
        C3_naiv
    )

    # --------------------------------------------------------
    # Differenzen
    # --------------------------------------------------------

    diff_greedy = V_opt - V_greedy
    diff_naiv = V_opt - V_naiv

    # --------------------------------------------------------
    # AUSGABE
    # --------------------------------------------------------

    print("\n================================================")
    print(f"σ = {sigma}")
    print("================================================")

    print(f"\nr1_opt     = {r1_opt:.6f}")
    print(f"r1_greedy  = {r1_greedy:.6f}")
    print(f"r1_naive   = {r1_naive:.6f}")

    print(f"\nV_opt      = {V_opt:.10f}")
    print(f"V_greedy   = {V_greedy:.10f}")
    print(f"V_naiv     = {V_naiv:.10f}")

    print(f"\nOpt-Greedy = {diff_greedy:.10f}")
    print(f"Opt-Naiv   = {diff_naiv:.10f}")

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(12, 7))

    # --------------------------------------------------------
    # p1
    # --------------------------------------------------------

    plt.plot(
        rs,
        p1_vals,
        linewidth=2,
        alpha=0.8,
        label=r"$p_1(r)$"
    )
    plt.plot(
    rs,
    p0_vals,
    linewidth=2,
    alpha=0.8,
    label=r"$p_0(r)$"
    )

    # --------------------------------------------------------
    # Gesamtwerte
    # --------------------------------------------------------

    plt.plot(
        rs,
        total_opt_vals,
        linewidth=3,
        label=r"$V_{\mathrm{opt}}(r)$"
    )

    plt.plot(
        rs,
        total_greedy_vals,
        linestyle='--',
        linewidth=3,
        label=r"$V_{\mathrm{greedy}}(r)$"
    )

    plt.plot(
        rs,
        total_naiv_vals,
        linestyle=':',
        linewidth=3,
        label=r"$V_{\mathrm{naiv}}(r)$"
    )

    # --------------------------------------------------------
    # Radien
    # --------------------------------------------------------

    plt.axvline(
        r1_opt,
        linewidth=2,
        label=rf"$r_1^*={r1_opt:.2f}$"
    )

    plt.axvline(
        r1_greedy,
        linestyle='--',
        linewidth=2,
        label=rf"$r_1^{{greedy}}={r1_greedy:.2f}$"
    )

    plt.axvline(
        r1_naive,
        linestyle=':',
        linewidth=2,
        label=rf"$r_1^{{naiv}}={r1_naive:.2f}$"
    )

    plt.xlabel("r (mm)")
    plt.ylabel("Wahrscheinlichkeit")

    plt.title(
        rf"Strategievergleich für $\sigma={sigma}$"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# SIGMA-STUDIE
# ============================================================

def sigma_study():

    sigmas = np.linspace(2, 60, 40)

    r1_opts = []
    r1_greeds = []

    vals_opt = []
    vals_greedy = []
    vals_naive = []

    diff_opt_greedy = []
    diff_opt_naive = []

    for sigma in sigmas:

        print(f"σ = {sigma:.2f}")

        rs = np.linspace(0, 200, 80)

        C2_opt = C2_optimal(sigma)
        C2_naiv = C2_naive(sigma)

        C3_opt = C3_optimal(sigma)
        C3_naiv = C3_naive(sigma)
        C3_greed = C3_greedy(sigma)

        total_opt_vals = np.array([
            total_value(r, sigma, C2_opt, C3_opt)
            for r in rs
        ])

        r1_opt = rs[np.argmax(total_opt_vals)]

        V_opt = np.max(total_opt_vals)

        r1_greed = greedy_r1(sigma)

        V_greedy = total_value(
            r1_greed,
            sigma,
            C2_opt,
            C3_greed
        )

        V_naiv = total_value(
            R1_NAIVE,
            sigma,
            C2_naiv,
            C3_naiv
        )

        r1_opts.append(r1_opt)
        r1_greeds.append(r1_greed)

        vals_opt.append(V_opt)
        vals_greedy.append(V_greedy)
        vals_naive.append(V_naiv)

        diff_opt_greedy.append(V_opt - V_greedy)
        diff_opt_naive.append(V_opt - V_naiv)

    # ========================================================
    # PLOT 1
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

    # ========================================================
    # PLOT 2
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        sigmas,
        vals_opt,
        linewidth=3,
        label="Optimal"
    )

    plt.plot(
        sigmas,
        vals_greedy,
        linestyle='--',
        linewidth=2,
        label="Greedy"
    )

    plt.plot(
        sigmas,
        vals_naive,
        linestyle=':',
        linewidth=2,
        label="Naiv"
    )

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Erfolgswahrscheinlichkeit")

    plt.title(
        "Strategievergleich 3-Punkte-Checkout mit 3 Darts relativ zu σ"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ========================================================
    # PLOT 3
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        sigmas,
        diff_opt_greedy,
        linewidth=3,
        label="Optimal - Greedy"
    )

    plt.plot(
        sigmas,
        diff_opt_naive,
        linewidth=3,
        label="Optimal - Naiv"
    )

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Absolute Differenz")

    plt.title(
        "Strategiedifferenzen relativ zu σ"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    plot_total_value_fixed_sigma(sigma=40)

    sigma_study()
