import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# ============================================================
# BOARD PARAMETER
# ============================================================
R_BULL_INNER = 6.35
R_BULL_OUTER = 15.9
R_TRIPLE_INNER = 99.0
R_TRIPLE_OUTER = 107.0
R_DOUBLE_INNER = 162.0
R_DOUBLE_OUTER = 170.0
segments = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5])
ANGLE_PER_SEG = 2 * np.pi / 20

def get_angle_for_segment_1():
    i = np.where(segments == 1)[0][0]
    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20

ANGLE_1 = get_angle_for_segment_1()
THETA_MIN_1 = ANGLE_1 - ANGLE_PER_SEG / 2
THETA_MAX_1 = ANGLE_1 + ANGLE_PER_SEG / 2

# ============================================================
# GAUSS-INTEGRAL + WAHRSCHEINLICHKEITEN
# ============================================================
def gaussian_polar_integrand(theta, rho, r0, theta0, sigma):
    return np.exp(-(rho**2 + r0**2 - 2 * rho * r0 * np.cos(theta - theta0)) / (2 * sigma**2))

def integrate_over_region(r_min, r_max, theta_min, theta_max, r0, theta0, sigma):
    def inner(theta, rho):
        return rho * gaussian_polar_integrand(theta, rho, r0, theta0, sigma)
    def radial(rho):
        v, _ = quad(inner, theta_min, theta_max, args=(rho,), epsabs=1e-7, epsrel=1e-7)
        return v
    val, _ = quad(radial, r_min, r_max, epsabs=1e-7, epsrel=1e-7)
    return val / (2 * np.pi * sigma**2)

def p_D1(r0, sigma):
    return integrate_over_region(R_DOUBLE_INNER, R_DOUBLE_OUTER, THETA_MIN_1, THETA_MAX_1, float(r0), ANGLE_1, sigma)

def p_single_1(r0, sigma):
    p_single = integrate_over_region(R_BULL_OUTER, R_DOUBLE_INNER, THETA_MIN_1, THETA_MAX_1, float(r0), ANGLE_1, sigma)
    p_triple = integrate_over_region(R_TRIPLE_INNER, R_TRIPLE_OUTER, THETA_MIN_1, THETA_MAX_1, float(r0), ANGLE_1, sigma)
    return p_single - p_triple

def p_miss(r0, sigma):
    def miss_int(rho):
        v, _ = quad(lambda th: rho * gaussian_polar_integrand(th, rho, float(r0), ANGLE_1, sigma), 0, 2*np.pi, epsabs=1e-6, epsrel=1e-6)
        return v
    val, _ = quad(miss_int, R_DOUBLE_OUTER, 1000, epsabs=1e-6, epsrel=1e-6)
    return val / (2 * np.pi * sigma**2)

# ============================================================
# 2-DART HELFER
# ============================================================
def find_maximum_pD1(sigma, bracket=(140, 190), tol=1e-8):
    def neg_pD1(r):
        return -p_D1(r, sigma)
    res = minimize_scalar(neg_pD1, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

# ============================================================
# results4 BERECHNUNG
# ============================================================
def create_results4():
    COMMON_SIGMAS = np.linspace(3, 60, 25)  # reduziert für Geschwindigkeit
    r_opt = []
    p_opt = []
    p_naive = []
    p_greedy = []
    print("=== Berechne results4 (Basis für 8-Punkte) ===\n")
    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        print(f"σ = {sigma:5.1f}")
        _, pD1_max = find_maximum_pD1(sigma)
        def neg_E(r1):
            return -(p_D1(r1, sigma) + (p_miss(r1, sigma) + p_single_1(r1, sigma)) * pD1_max)
        res = minimize_scalar(neg_E, bounds=(120, 200), tol=1e-8, method='bounded')
        r_opt.append(res.x)
        p_opt.append(-res.fun)
        # Naive & Greedy
        r_naive = 166.0
        p_naive_val = p_D1(r_naive, sigma) + (p_miss(r_naive, sigma) + p_single_1(r_naive, sigma)) * p_D1(r_naive, sigma)
        p_greedy_val = p_D1(r_naive, sigma) + (p_miss(r_naive, sigma) + p_single_1(r_naive, sigma)) * pD1_max
        p_naive.append(p_naive_val)
        p_greedy.append(p_greedy_val)
    return {
        'sigma': COMMON_SIGMAS,
        'p_opt': np.array(p_opt),
        'p_naive': np.array(p_naive),
        'p_greedy': np.array(p_greedy)
    }

results4 = create_results4()

def get_p_from_results4(sigma, key='p_opt'):
    interp = interp1d(results4['sigma'], results4[key], kind='linear', fill_value="extrapolate")
    return float(interp(sigma))

# ============================================================
# 8-PUNKTE CHECKOUT FUNKTIONEN (mit results4)
# ============================================================
def find_maximum_E_8points(sigma, bracket=(120, 200), tol=1e-8):
    pD1_max = get_p_from_results4(sigma, 'p_opt')
    def neg_E(r1):
        p_d1 = p_D1(r1, sigma)
        p1 = p_single_1(r1, sigma)
        p0 = p_miss(r1, sigma)
        E = p_d1 + (p0 + p1) * pD1_max
        return -E
    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

def checkout_8points_naive(sigma, r_naive=166.0):
    p_d1_1 = p_D1(r_naive, sigma)
    p1 = p_single_1(r_naive, sigma)
    p0 = p_miss(r_naive, sigma)
    p_d1_2 = get_p_from_results4(sigma, 'p_naive')
    E = p_d1_1 + (p0 + p1) * p_d1_2
    return E, r_naive, r_naive

def checkout_8points_greedy(sigma):
    r_greedy, _ = find_maximum_pD1(sigma)
    p_d1_1 = p_D1(r_greedy, sigma)
    p1 = p_single_1(r_greedy, sigma)
    p0 = p_miss(r_greedy, sigma)
    p_d1_2 = get_p_from_results4(sigma, 'p_greedy')
    E = p_d1_1 + (p0 + p1) * p_d1_2
    return E, r_greedy, r_greedy

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# (BOARD + INTEGRAL + p_D1, p_single_1, p_miss, find_maximum_pD1 unverändert – wie oben)

# ... [Vorheriger Code bis create_results4() und get_p_from_results4() bleibt gleich] ...

# ============================================================
# VERGLEICH FÜR 8-PUNKTE
# ============================================================
def checkout_comparison_8points():
    COMMON_SIGMAS = np.linspace(3, 60, 25)
    r_opt_list = []
    r_greedy_list = []
    p_opt_list = []
    p_naive_list = []
    p_greedy_list = []
    print("=== 8-Punkte Checkout Vergleich ===\n")
    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        print(f"σ = {sigma:5.1f}")
        # Optimal
        r1_opt, pE_opt = find_maximum_E_8points(sigma)
        # Naive
        pE_naive, _, _ = checkout_8points_naive(sigma)
        # Greedy
        pE_greedy, r_greedy, _ = checkout_8points_greedy(sigma)
        r_opt_list.append(r1_opt)
        r_greedy_list.append(r_greedy)
        p_opt_list.append(pE_opt)
        p_naive_list.append(pE_naive)
        p_greedy_list.append(pE_greedy)
    return {
        'sigma': COMMON_SIGMAS,
        'r_opt': np.array(r_opt_list),
        'r_greedy': np.array(r_greedy_list),
        'p_opt': np.array(p_opt_list),
        'p_naive': np.array(p_naive_list),
        'p_greedy': np.array(p_greedy_list)
    }

# ============================================================
# PLOTS
# ============================================================
def plot_8points_comparison(results):
    sig = results['sigma']
    # 1. Wahrscheinlichkeiten
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("8-Punkte-Checkout mit 2 Darts – Strategie-Vergleich", fontsize=16)
    axs[0].plot(sig, results['p_opt'], 'o-', lw=2.5, label='Optimal (angepasstes r1)')
    axs[0].plot(sig, results['p_naive'], 's--', lw=2, label='Naiv (r=166 mm)')
    axs[0].plot(sig, results['p_greedy'], '^-.', lw=2, label='Greedy (r=r_max_D1)')
    axs[0].set_ylabel("Checkout-Wahrscheinlichkeit")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    # Differenzen
    axs[1].plot(sig, results['p_opt'] - results['p_naive'], 'o-', lw=2, label='Optimal - Naiv', color='purple')
    axs[1].plot(sig, results['p_opt'] - results['p_greedy'], 's--', lw=2, label='Optimal - Greedy', color='green')
    axs[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"σ (mm)")
    axs[1].set_ylabel("Differenz")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    plt.tight_layout()
   
    # 2. Optimale Radien
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Optimale Zielradien r1 für 8-Punkte-Checkout mit 2 Darts", fontsize=16)
    ax.plot(sig, results['r_opt'], 'o-', lw=2.5, label='r1_opt (Optimal)', color='darkblue')
    ax.plot(sig, [166.0]*len(sig), 's--', lw=2, label='r_naive = 166 mm', color='darkorange')
    ax.plot(sig, results['r_greedy'], 'd-.', lw=2, label='r_greedy = r_max_D1', color='green')
    ax.set_xlabel(r"σ (mm)")
    ax.set_ylabel("Radius r1 (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results8 = checkout_comparison_8points()
    plot_8points_comparison(results8)
    print("\nFertig! Plots erzeugt.")
