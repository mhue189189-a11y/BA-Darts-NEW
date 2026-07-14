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

segments = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                     3, 19, 7, 16, 8, 11, 14, 9, 12, 5])
ANGLE_PER_SEG = 2 * np.pi / 20

def get_angle_for_segment_1():
    i = np.where(segments == 1)[0][0]
    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20

ANGLE_1 = get_angle_for_segment_1()
THETA_MIN_1 = ANGLE_1 - ANGLE_PER_SEG / 2
THETA_MAX_1 = ANGLE_1 + ANGLE_PER_SEG / 2

# ============================================================
# KERN-FUNKTIONEN (GAUSS-INTEGRAL)
# ============================================================
def gaussian_polar_integrand(theta, rho, r0, theta0, sigma):
    return np.exp(
        -(rho**2 + r0**2 - 2 * rho * r0 * np.cos(theta - theta0)) / (2 * sigma**2)
    )

def integrate_over_region(r_min, r_max, theta_min, theta_max, r0, theta0, sigma):
    def inner(theta, rho):
        return rho * gaussian_polar_integrand(theta, rho, r0, theta0, sigma)
    def radial(rho):
        v, _ = quad(inner, theta_min, theta_max, args=(rho,), epsabs=1e-7, epsrel=1e-7)
        return v
    val, _ = quad(radial, r_min, r_max, epsabs=1e-7, epsrel=1e-7)
    return val / (2 * np.pi * sigma**2)

# ============================================================
# WAHRSCHEINLICHKEITEN
# ============================================================
def p_D1(r0, sigma):
    r0 = float(r0)
    return integrate_over_region(R_DOUBLE_INNER, R_DOUBLE_OUTER, THETA_MIN_1, THETA_MAX_1, r0, ANGLE_1, sigma)

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
        v, _ = quad(lambda th: rho * gaussian_polar_integrand(th, rho, r0, ANGLE_1, sigma), 0, 2*np.pi)
        return v
    val, _ = quad(miss_int, R_DOUBLE_OUTER, 1000, epsabs=1e-6)
    return val / (2 * np.pi * sigma**2)

# ============================================================
# 4-PUNKTE MIT 2 DARTS (C4 BERECHNUNG)
# ============================================================
def checkout_probability_4points(r1, r2, sigma):
    p_d1_1 = p_D1(r1, sigma)
    p1_1 = p_single_1(r1, sigma)
    p0_1 = p_miss(r1, sigma)
    p_d1_2 = p_D1(r2, sigma)
    return p_d1_1 + (p0_1 + p1_1) * p_d1_2

def find_maximum_pD1(sigma, bracket=(130, 220), tol=1e-8):
    def neg_pD1(r):
        return -p_D1(r, sigma)
    res = minimize_scalar(neg_pD1, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

def find_maximum_E_4points(sigma, bracket=(130, 220), tol=1e-8):
    r2_opt, pD1_max = find_maximum_pD1(sigma, bracket=bracket, tol=tol)
    def neg_E(r1):
        return -checkout_probability_4points(r1, r2_opt, sigma)
    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    r1_opt = res.x
    E_opt = -res.fun
    return r1_opt, r2_opt, E_opt, pD1_max

# ============================================================
# COMMON SIGMAS
# ============================================================
COMMON_SIGMAS = np.linspace(3, 60, 30)

# ============================================================
# RUN 4-PUNKTE / 2-DARTS STRATEGIE (C4)
# ============================================================
def checkout_comparison_4points():
    r1_opt_list, r2_opt_list = [], []
    p_opt = []
    
    print("=== Berechne C4 (4-Punkte mit 2 Darts) ===\n")
    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        r1_opt, r2_opt, pE_opt, _ = find_maximum_E_4points(sigma)
        r1_opt_list.append(r1_opt)
        r2_opt_list.append(r2_opt)
        p_opt.append(pE_opt)
        
    resultsC4 = {
        'sigma': COMMON_SIGMAS.copy(),
        'r1_opt': np.array(r1_opt_list),
        'r2_opt': np.array(r2_opt_list),
        'p_opt': np.array(p_opt)
    }
    return resultsC4

# ============================================================
# OPTIMIERTES 3-DARTS STRATEGIESPIEL (ERWARTUNGSWERT)
# ============================================================
def find_max_E_3darts_adapted(sigma, resultsC4, bracket=(130, 220), tol=1e-8):
    C4 = np.interp(sigma, resultsC4['sigma'], resultsC4['p_opt'])
    
    def neg_E(r):
        p_d1 = p_D1(r, sigma)
        p0 = p_miss(r, sigma)
        p1 = p_single_1(r, sigma)
        # E = pD1 + (p0 + p1) * C4
        return -(p_d1 + (p0 + p1) * C4)
        
    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

# ============================================================
# COMPONENT FOR 3-DARTS SIMULATION
# ============================================================
def checkout_comparison_3darts_adapted(resultsC4):
    r_opt_3 = []
    r_naive_3 = []
    r_greedy_3 = []
    p_opt_3 = []
    p_naive_3 = []
    p_greedy_3 = []
    
    print("\n=== Berechne optimierte 3-Darts Checkout-Kurven ===\n")
    
    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        print(f"σ = {sigma:5.1f}")
        
        # 1. Optimierter Radius (hier ist r1 nun deutlich freier nach innen verschiebbar)
        r1_opt, pE_opt = find_max_E_3darts_adapted(sigma, resultsC4)
        
        # C4 für dieses Sigma holen
        C4 = np.interp(sigma, resultsC4['sigma'], resultsC4['p_opt'])
        
        # 2. Naiver Radius (immer 166.0 mm)
        r_naiv = 166.0
        p_d1_n = p_D1(r_naiv, sigma)
        p0_n = p_miss(r_naiv, sigma)
        p1_n = p_single_1(r_naiv, sigma)
        pE_naiv = p_d1_n + (p0_n + p1_n) * C4
        
        # 3. Greedy Radius (Zielen direkt auf das Maximum der Double 1)
        r_d1_g, _ = find_maximum_pD1(sigma)
        p_d1_g = p_D1(r_d1_g, sigma)
        p0_g = p_miss(r_d1_g, sigma)
        p1_g = p_single_1(r_d1_g, sigma)
        pE_greedy = p_d1_g + (p0_g + p1_g) * C4
        
        r_opt_3.append(r1_opt)
        r_naive_3.append(r_naiv)
        r_greedy_3.append(r_d1_g)
        p_opt_3.append(pE_opt)
        p_naive_3.append(pE_naiv)
        p_greedy_3.append(pE_greedy)
        
    resultsC3 = {
        'sigma': COMMON_SIGMAS.copy(),
        'r_opt_3': np.array(r_opt_3),
        'p_opt_3': np.array(p_opt_3),
        'r_naive_3': np.array(r_naive_3),
        'p_naive_3': np.array(p_naive_3),
        'r_greedy_3': np.array(r_greedy_3),
        'p_greedy_3': np.array(p_greedy_3)
    }
    return resultsC3

# ============================================================
# PLOTS
# ============================================================
def plot_results(resultsC4, resultsC3):
    sig = resultsC3['sigma']
    
    # Plot der Wahrscheinlichkeiten
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("3-Darts Checkout (Optimiert mit C4) - Strategievergleich", fontsize=16)
    
    axs[0].plot(sig, resultsC3['p_opt_3'], 'o-', lw=2.5, label='Optimiert (Dart 1)', color='darkblue')
    axs[0].plot(sig, resultsC3['p_naive_3'], 's--', lw=2, label='Naiv (166mm)', color='darkorange')
    axs[0].plot(sig, resultsC3['p_greedy_3'], '^-.', lw=2, label='Greedy (Max p_D1)', color='green')
    axs[0].set_ylabel("Checkout-Wahrscheinlichkeit")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    diff_opt_naiv = resultsC3['p_opt_3'] - resultsC3['p_naive_3']
    diff_opt_greedy = resultsC3['p_opt_3'] - resultsC3['p_greedy_3']
    axs[1].plot(sig, diff_opt_naiv, 'o-', lw=2, label='opt - naiv', color='purple')
    axs[1].plot(sig, diff_opt_greedy, 's--', lw=2, label='opt - greedy (sollte fast 0 sein)', color='brown')
    axs[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"$\sigma$ (mm)")
    axs[1].set_ylabel("Differenz der Wahrscheinlichkeiten")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

    # Plot der Radien
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Vergleich der Zielradien für den 7 Punke Checkout mit 3 Darts", fontsize=16)
    ax.plot(sig, resultsC3['r_opt_3'], 'o-', lw=2.5, label='r1_opt (3-Darts optimiert)', color='darkblue')
    ax.plot(sig, resultsC4['r1_opt'], 's--', lw=2, label='r1_opt (C4 Vergleich)', color='darkorange')
    ax.plot(sig, resultsC3['r_greedy_3'], 'd-.', lw=2, label='r_greedy (Max p_D1)', color='green')
    ax.set_xlabel(r"$\sigma$ (mm)")
    ax.set_ylabel("Aim-Radius r (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    resultsC4 = checkout_comparison_4points()
    resultsC3 = checkout_comparison_3darts_adapted(resultsC4)
    plot_results(resultsC4, resultsC3)
    print("\nAlle Plots erzeugt!")
