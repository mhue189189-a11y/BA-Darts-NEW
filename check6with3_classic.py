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

def find_maximum_pD1(sigma, bracket=(140, 190), tol=1e-8):
    def neg_pD1(r):
        return -p_D1(r, sigma)
    res = minimize_scalar(neg_pD1, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

# ============================================================
# VORBERECHNUNG: C2 & C3
# ============================================================
def calculate_C2_and_C3(sigmas):
    c2_opt, c2_naive = [], []
    c3_opt, c3_naive = [], []
    
    r2_opt_list, r3_opt_list = [], []
    
    print("=== Berechne Basis-Wahrscheinlichkeiten (C2 & C3) ===")
    for sigma in sigmas:
        sigma = float(sigma)
        r_greedy, pD1_max = find_maximum_pD1(sigma)
        
        # --- C2 (2 Punkte mit 2 Darts) ---
        def neg_E_c2(r):
            return -(p_D1(r, sigma) + p_miss(r, sigma) * pD1_max)
        res_c2 = minimize_scalar(neg_E_c2, bounds=(130, 200), tol=1e-8, method='bounded')
        
        c2_opt.append(-res_c2.fun)
        r2_opt_list.append(res_c2.x)
        r3_opt_list.append(r_greedy)
        
        r_naiv = 166.0
        c2_naive.append(p_D1(r_naiv, sigma) + p_miss(r_naiv, sigma) * p_D1(r_naiv, sigma))
        
        # --- C3 (3 Punkte mit 2 Darts) ---
        def neg_E_c3(r):
            return -(p_single_1(r, sigma) * (-res_c2.fun))
        res_c3 = minimize_scalar(neg_E_c3, bounds=(110, 170), tol=1e-8, method='bounded')
        
        c3_opt.append(-res_c3.fun)
        c3_naive.append(p_single_1(134.5, sigma) * c2_naive[-1]) # 134.5 mm = Mitte großes S1

    return {
        'sigma': sigmas,
        'c2_opt': np.array(c2_opt), 'c2_naive': np.array(c2_naive),
        'c3_opt': np.array(c3_opt), 'c3_naive': np.array(c3_naive),
        'r2_opt': np.array(r2_opt_list), 'r3_opt': np.array(r3_opt_list)
    }

# ============================================================
# HAUPTBERECHNUNG: 6-PUNKTE CHECKOUT MIT 3 DARTS
# ============================================================
def checkout_comparison_6points():
    COMMON_SIGMAS = np.linspace(3, 60, 25)
    base = calculate_C2_and_C3(COMMON_SIGMAS)
    
    # Interpolationen für C2 und C3
    get_c2_opt = interp1d(base['sigma'], base['c2_opt'], kind='linear', fill_value="extrapolate")
    get_c2_naive = interp1d(base['sigma'], base['c2_naive'], kind='linear', fill_value="extrapolate")
    
    get_c3_opt = interp1d(base['sigma'], base['c3_opt'], kind='linear', fill_value="extrapolate")
    get_c3_naive = interp1d(base['sigma'], base['c3_naive'], kind='linear', fill_value="extrapolate")
    
    get_r2_opt = interp1d(base['sigma'], base['r2_opt'], kind='linear', fill_value="extrapolate")
    get_r3_opt = interp1d(base['sigma'], base['r3_opt'], kind='linear', fill_value="extrapolate")

    r1_opt_list = []
    r2_opt_list = []
    r3_opt_list = []
    
    p_opt_list = []
    p_naive_list = []
    p_greedy_list = []
    
    print("\n=== Berechne 6-Punkte Checkout mit 3 Darts ===")
    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        print(f"σ = {sigma:5.1f}")
        
        c2_opt_val = float(get_c2_opt(sigma))
        c2_naive_val = float(get_c2_naive(sigma))
        c3_opt_val = float(get_c3_opt(sigma))
        c3_naive_val = float(get_c3_naive(sigma))
        
        # 1. OPTIMIERUNG FÜR DART 1 (r1_opt)
        def neg_E(r1):
            p_d = p_D1(r1, sigma)
            p1 = p_single_1(r1, sigma)
            p0 = p_miss(r1, sigma)
            return -(p_d + p1 * c3_opt_val + p0 * c2_opt_val)
            
        res = minimize_scalar(neg_E, bounds=(110, 200), tol=1e-8, method='bounded')
        r1_opt = res.x
        
        r1_opt_list.append(r1_opt)
        r2_opt_list.append(float(get_r2_opt(sigma)))
        r3_opt_list.append(float(get_r3_opt(sigma)))
        
        # 2. STRATEGIEBEWERTUNG
        # Optimal (Beste Radienkette)
        p_opt_list.append(-res.fun)
        
        # Naiv (Alle Darts stur auf 166.0 mm)
        r_naiv = 166.0
        p_naive_list.append(p_D1(r_naiv, sigma) + p_single_1(r_naiv, sigma) * c3_naive_val + p_miss(r_naiv, sigma) * c2_naive_val)
        
        # Greedy (Erster Dart auf gieriges r_greedy, danach optimale Fortführung C2/C3)
        r_greedy, _ = find_maximum_pD1(sigma)
        p_greedy_list.append(p_D1(r_greedy, sigma) + p_single_1(r_greedy, sigma) * c3_opt_val + p_miss(r_greedy, sigma) * c2_opt_val)

    return {
        'sigma': COMMON_SIGMAS,
        'r1_opt': np.array(r1_opt_list),
        'r2_opt': np.array(r2_opt_list),
        'r3_opt': np.array(r3_opt_list),
        'p_opt': np.array(p_opt_list),
        'p_naive': np.array(p_naive_list),
        'p_greedy': np.array(p_greedy_list)
    }

# ============================================================
# PLOTS
# ============================================================
def plot_6points_results(results):
    sig = results['sigma']
    
    # 1. Strategievergleich (Wahrscheinlichkeiten)
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("6-Punkte-Checkout mit 3 Darts – Strategie-Vergleich", fontsize=16)
    
    axs[0].plot(sig, results['p_opt'], 'o-', lw=2.5, label='Optimal', color='darkblue')
    axs[0].plot(sig, results['p_naive'], 's--', lw=2, label='Naiv (r=166 mm)', color='darkorange')
    axs[0].plot(sig, results['p_greedy'], '^-.', lw=2, label='Greedy ', color='green')
    axs[0].set_ylabel("Checkout-Wahrscheinlichkeit")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Differenzen
    axs[1].plot(sig, results['p_opt'] - results['p_naive'], 'o-', lw=2, label='Optimal - Naiv', color='purple')
    axs[1].plot(sig, results['p_opt'] - results['p_greedy'], 's--', lw=2, label='Optimal - Greedy', color='brown')
    axs[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"σ (mm)")
    axs[1].set_ylabel("Gewinn an Checkout-W-Keit")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Radiusplot (Alle 3 Darts der Optimalstrategie im Vergleich)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Optimale Zielradien für die 3 Darts (6-Punkte-Checkout)", fontsize=16)
    
    ax.plot(sig, results['r1_opt'], 'o-', lw=2.5, label='r1_opt ', color='darkblue')
    ax.plot(sig, results['r3_opt'], 'd-.', lw=1.5, label='r3_opt', color='forestgreen')
    
    # Referenzen
    ax.axhline(166.0, color='darkorange', linestyle=':', lw=2, label='r_naiv (166 mm)')
    
    ax.set_xlabel(r"σ (mm)")
    ax.set_ylabel("Zielradius r (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results6 = checkout_comparison_6points()
    plot_6points_results(results6)
    print("\nFertig! Alle Plots erzeugt.")
