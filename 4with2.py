import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# ============================================================
# BOARD PARAMETER
# ============================================================
R_BULL_INNER = 6.35
R_BULL_OUTER = 15.9
R_TRIPLE_INNER = 99.0
R_TRIPLE_OUTER = 107.0
R_DOUBLE_INNER = 162.0
R_DOUBLE_OUTER = 170.0

segments = np.array([
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
])

ANGLE_PER_SEG = 2 * np.pi / 20

def get_angle_for_segment_1():
    i = np.where(segments == 1)[0][0]
    return np.pi / 2 - (i + 0.5) * ANGLE_PER_SEG + np.pi / 20

ANGLE_1 = get_angle_for_segment_1()
THETA_MIN_1 = ANGLE_1 - ANGLE_PER_SEG / 2
THETA_MAX_1 = ANGLE_1 + ANGLE_PER_SEG / 2

# ============================================================
# GAUSS-INTEGRAL
# ============================================================
def gaussian_polar_integrand(theta, rho, r0, theta0, sigma):
    return np.exp(
        -(rho**2 + r0**2 - 2 * rho * r0 * np.cos(theta - theta0)) / (2 * sigma**2)
    )

def integrate_over_region(r_min, r_max, theta_min, theta_max, r0, theta0, sigma):
    def inner(theta, rho):
        return rho * gaussian_polar_integrand(theta, rho, r0, theta0, sigma)

    def radial(rho):
        v, _ = quad(
            inner,
            theta_min,
            theta_max,
            args=(rho,),
            epsabs=1e-7,
            epsrel=1e-7
        )
        return v

    val, _ = quad(
        radial,
        r_min,
        r_max,
        epsabs=1e-7,
        epsrel=1e-7
    )
    return val / (2 * np.pi * sigma**2)

# ============================================================
# WAHRSCHEINLICHKEITEN
# ============================================================
def p_D1(r0, sigma):
    r0 = float(r0)
    return integrate_over_region(
        R_DOUBLE_INNER,
        R_DOUBLE_OUTER,
        THETA_MIN_1,
        THETA_MAX_1,
        r0,
        ANGLE_1,
        sigma
    )

def p_single_1(r0, sigma):
    r0 = float(r0)
    p_single = integrate_over_region(
        R_BULL_OUTER,
        R_DOUBLE_INNER,
        THETA_MIN_1,
        THETA_MAX_1,
        r0,
        ANGLE_1,
        sigma
    )
    p_triple = integrate_over_region(
        R_TRIPLE_INNER,
        R_TRIPLE_OUTER,
        THETA_MIN_1,
        THETA_MAX_1,
        r0,
        ANGLE_1,
        sigma
    )
    return p_single - p_triple

def p_miss(r0, sigma):
    r0 = float(r0)

    def miss_int(rho):
        v, _ = quad(
            lambda th: rho * gaussian_polar_integrand(th, rho, r0, ANGLE_1, sigma),
            0,
            2 * np.pi,
            epsabs=1e-6,
            epsrel=1e-6
        )
        return v

    val, _ = quad(
        miss_int,
        R_DOUBLE_OUTER,
        1000,
        epsabs=1e-6,
        epsrel=1e-6
    )
    return val / (2 * np.pi * sigma**2)

# ============================================================
# CHECKOUT-WAHRSCHEINLICHKEIT (DIREKT AUS DEN RADIE)
# ============================================================
def checkout_probability_4points(r1, r2, sigma):
    p_d1_1 = p_D1(r1, sigma)
    p1_1 = p_single_1(r1, sigma)
    p0_1 = p_miss(r1, sigma)
    p_d1_2 = p_D1(r2, sigma)
    return p_d1_1 + (p0_1 + p1_1) * p_d1_2

# ============================================================
# OPTIMIERUNGEN & STRATEGIEN
# ============================================================
def find_maximum_pD1(sigma, bracket=(140, 190), tol=1e-8):
    def neg_pD1(r):
        return -p_D1(r, sigma)

    res = minimize_scalar(neg_pD1, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

def find_maximum_E_4points(sigma, bracket=(120, 200), tol=1e-8):
    r2_opt, pD1_max = find_maximum_pD1(sigma, bracket=bracket, tol=tol)

    def neg_E(r1):
        return -checkout_probability_4points(r1, r2_opt, sigma)

    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    r1_opt = res.x
    E_opt = -res.fun
    return r1_opt, r2_opt, E_opt, pD1_max

def checkout_4points_naive(sigma, r_naive=166.0):
    # Beide Darts werfen stur auf die Mitte des Double-Rings (166 mm)
    E = checkout_probability_4points(r_naive, r_naive, sigma)
    return E, r_naive, r_naive

def checkout_4points_greedy(sigma):
    # Beide Darts zielen auf den optimalen Einzel-Double-Radius r_greedy
    r_greedy, _ = find_maximum_pD1(sigma)
    E = checkout_probability_4points(r_greedy, r_greedy, sigma)
    return E, r_greedy, r_greedy

# ============================================================
# VERGLEICH ÜBER SIGMAS
# ============================================================
COMMON_SIGMAS = np.linspace(3, 60, 30)

def checkout_comparison_4points():
    r1_opt_list = []
    r2_opt_list = []
    r1_naive_list = []
    r2_naive_list = []
    r1_greedy_list = []
    r2_greedy_list = []

    p_opt = []
    p_naive = []
    p_greedy = []

    print("=== 4-Punkte Checkout mit 2 Darts ===\n")

    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        print(f"σ = {sigma:5.1f}")

        # Berechnung der Strategien aus ihren jeweiligen Radien
        r1_opt, r2_opt, pE_opt, _ = find_maximum_E_4points(sigma)
        pE_naiv, r1_naiv, r2_naiv = checkout_4points_naive(sigma)
        pE_g, r1_g, r2_g = checkout_4points_greedy(sigma)

        r1_opt_list.append(r1_opt)
        r2_opt_list.append(r2_opt)
        r1_naive_list.append(r1_naiv)
        r2_naive_list.append(r2_naiv)
        r1_greedy_list.append(r1_g)
        r2_greedy_list.append(r2_g)

        p_opt.append(pE_opt)
        p_naive.append(pE_naiv)
        p_greedy.append(pE_g)

    results = {
        'sigma': COMMON_SIGMAS.copy(),
        'r1_opt': np.array(r1_opt_list),
        'r2_opt': np.array(r2_opt_list),
        'r1_naive': np.array(r1_naive_list),
        'r2_naive': np.array(r2_naive_list),
        'r1_greedy': np.array(r1_greedy_list),
        'r2_greedy': np.array(r2_greedy_list),
        'p_opt': np.array(p_opt),
        'p_naive': np.array(p_naive),
        'p_greedy': np.array(p_greedy)
    }
    return results

# ============================================================
# PLOTS
# ============================================================
def plot_comparison_4points(results):
    sig = results['sigma']

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("4-Punkte-Checkout mit 2 Darts - Strategie-Vergleich", fontsize=16)

    axs[0].plot(sig, results['p_opt'], 'o-', lw=2.5, label='Optimal')
    axs[0].plot(sig, results['p_naive'], 's--', lw=2, label='Naiv')
    axs[0].plot(sig, results['p_greedy'], '^-.', lw=2, label='Greedy')
    axs[0].set_ylabel("Checkout-Wahrscheinlichkeit")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(sig, results['p_opt'] - results['p_naive'], 'o-', lw=2, label='opt - naiv', color='purple')
    axs[1].plot(sig, results['p_opt'] - results['p_greedy'], 's--', lw=2, label='opt - greedy', color='green')
    axs[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"σ (mm)")
    axs[1].set_ylabel("Differenz")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Zielradien für das 4-Punkte-Checkout mit 2 Darts", fontsize=16)

    ax.plot(sig, results['r1_opt'], 'o-', lw=2.5, label='r1_opt (1. Dart Optimal)', color='darkblue')
    ax.plot(sig, results['r2_opt'], 'o--', lw=2.5, label='r2_opt (2. Dart Optimal)', color='navy')
    ax.plot(sig, results['r1_naive'], 's--', lw=2, label='r_naive = 166 mm', color='darkorange')
    ax.plot(sig, results['r1_greedy'], 'd-.', lw=2, label='r_greedy (Maximum p_D1)', color='green')

    ax.set_xlabel(r"σ (mm)")
    ax.set_ylabel("Radius r (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results4 = checkout_comparison_4points()
    plot_comparison_4points(results4)
    print("\nAlle Plots erzeugt!")
