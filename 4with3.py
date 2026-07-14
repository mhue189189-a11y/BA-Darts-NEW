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

def get_angle_for_segment(seg_value):
    i = np.where(segments == seg_value)[0][0]
    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20

ANGLE_1 = get_angle_for_segment(1)
ANGLE_2 = get_angle_for_segment(2)

THETA_MIN_1 = ANGLE_1 - ANGLE_PER_SEG / 2
THETA_MAX_1 = ANGLE_1 + ANGLE_PER_SEG / 2

THETA_MIN_2 = ANGLE_2 - ANGLE_PER_SEG / 2
THETA_MAX_2 = ANGLE_2 + ANGLE_PER_SEG / 2

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
# PROBABILITIES FÜR SEGMENT 1 / SEGMENT 2 / DOUBLE 1
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

def p_single_2(r0, sigma):
    r0 = float(r0)
    return integrate_over_region(
        R_BULL_OUTER,
        R_DOUBLE_INNER,
        THETA_MIN_2,
        THETA_MAX_2,
        r0,
        ANGLE_2,
        sigma
    ) - integrate_over_region(
        R_TRIPLE_INNER,
        R_TRIPLE_OUTER,
        THETA_MIN_2,
        THETA_MAX_2,
        r0,
        ANGLE_2,
        sigma
    )

# ============================================================
# DIRETKTE BERECHNUNG DER BEGLEIT-ZUSTÄNDE (E-WERTE AUS RADIE)
# ============================================================
def find_maximum_pD1(sigma, bracket=(140, 190), tol=1e-8):
    def neg_pD1(r):
        return -p_D1(r, sigma)

    res = minimize_scalar(neg_pD1, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

def find_maximum_E_2darts(sigma, bracket=(140, 190), tol=1e-8):
    """Berechnet das Optimum für 2 Darts auf 2 Restpunkte (C2)"""
    r_d1, p_max_d1 = find_maximum_pD1(sigma, bracket, tol)

    def neg_E(r):
        return -(p_D1(r, sigma) + p_miss(r, sigma) * p_max_d1)

    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun, r_d1, p_max_d1

def find_maximum_E_4points_2darts(sigma, bracket=(120, 200), tol=1e-8):
    """Berechnet das Optimum für 2 Darts auf 4 Restpunkte"""
    r2_greedy, pD1_max = find_maximum_pD1(sigma, bracket=bracket, tol=tol)

    def neg_E(r1):
        p_d1 = p_D1(r1, sigma)
        p1 = p_single_1(r1, sigma)
        p0 = p_miss(r1, sigma)
        E = p_d1 + (p0 + p1) * pD1_max
        return -E

    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    r1_opt = res.x
    E_opt = -res.fun
    return r1_opt, r2_greedy, E_opt, pD1_max

# ============================================================
# OPTIMALE INTEGRATION FÜR 3 DARTS
# ============================================================
def find_maximum_E_4points_3darts(sigma, results4, C2, bracket=(120, 200), tol=1e-8):
    res4 = np.interp(sigma, results4["sigma"], results4["p_opt"])
    c2 = np.interp(sigma, C2["sigma"], C2["p_opt"])

    def neg_E(r1):
        p_d1 = p_D1(r1, sigma)
        p0 = p_miss(r1, sigma)
        p2 = p_single_2(r1, sigma)
        E = p_d1 + p0 * res4 + p2 * c2
        return -E

    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

# ============================================================
# MAIN BERECHNUNGSPIPELINE
# ============================================================
def compute_resultsC2():
    sigmas = np.linspace(1, 60, 30)
    r_opt = []
    p_opt = []
    
    for sigma in sigmas:
        r1_opt, pE_opt, _, _ = find_maximum_E_2darts(sigma)
        r_opt.append(r1_opt)
        p_opt.append(pE_opt)

    return {
        "sigma": sigmas,
        "r_opt": np.array(r_opt),
        "p_opt": np.array(p_opt),
    }

def compute_result4():
    sigmas = np.linspace(3, 60, 30)
    r_opt = []
    p_opt = []

    for sigma in sigmas:
        r1_opt, _, pE_opt, _ = find_maximum_E_4points_2darts(sigma)
        r_opt.append(r1_opt)
        p_opt.append(pE_opt)

    return {
        "sigma": sigmas,
        "r_opt": np.array(r_opt),
        "p_opt": np.array(p_opt),
    }

def build_3dart_results(results4, C2):
    sigmas = results4["sigma"]
    
    # Listen initialisieren
    r1_opt, r2_opt, r3_opt, p_opt = [], [], [], []
    r1_naive, r2_naive, r3_naive, p_naive = [], [], [], []
    r1_greedy, r2_greedy, r3_greedy, p_greedy = [], [], [], []

    for sigma in sigmas:
        sigma = float(sigma)
        
        # 1) OPTIMAL (Nutzt die optimierten Interpolationsstützen der Vorstufen)
        r1, pE, = find_maximum_E_4points_3darts(sigma, results4, C2)
        r2_old, r3_old, _, _ = find_maximum_E_4points_2darts(sigma)
        
        r1_opt.append(r1)
        r2_opt.append(r2_old)
        r3_opt.append(r3_old)
        p_opt.append(pE)

        # 2) NAIV (Alle 3 Darts werfen auf das Double-Zentrum r = 166.0)
        rn = 166.0
        # Erwartungswerte für Folgesituationen direkt aus rn berechnen:
        # C2 mit 2 Darts komplett naiv (r = 166)
        c2_naiv = p_D1(rn, sigma) + p_miss(rn, sigma) * p_D1(rn, sigma)
        # 4-Punkte-Rest mit 2 Darts naiv (r = 166)
        res4_naiv = p_D1(rn, sigma) + (p_miss(rn, sigma) + p_single_1(rn, sigma)) * p_D1(rn, sigma)
        
        # Gesamterwartungswert für Dart 1 auf naiv r = 166:
        p_naive_val = p_D1(rn, sigma) + p_miss(rn, sigma) * res4_naiv + p_single_2(rn, sigma) * c2_naiv
        
        r1_naive.append(rn)
        r2_naive.append(rn)
        r3_naive.append(rn)
        p_naive.append(p_naive_val)

        # 3) GREEDY (Alle 3 Darts zielen auf das jeweilige Maximum der Double-Wahrscheinlichkeit r_g)
        rg, pD1_max = find_maximum_pD1(sigma)
        # C2 mit 2 Darts komplett greedy (r = rg)
        c2_greedy = p_D1(rg, sigma) + p_miss(rg, sigma) * pD1_max
        # 4-Punkte-Rest mit 2 Darts greedy (r = rg)
        res4_greedy = p_D1(rg, sigma) + (p_miss(rg, sigma) + p_single_1(rg, sigma)) * pD1_max
        
        # Gesamterwartungswert für Dart 1 auf greedy r = rg:
        p_greedy_val = p_D1(rg, sigma) + p_miss(rg, sigma) * res4_greedy + p_single_2(rg, sigma) * c2_greedy
        
        r1_greedy.append(rg)
        r2_greedy.append(rg)
        r3_greedy.append(rg)
        p_greedy.append(p_greedy_val)

    return {
        "sigma": sigmas,
        "r1_opt": np.array(r1_opt),
        "r2_opt": np.array(r2_opt),
        "r3_opt": np.array(r3_opt),
        "p_opt": np.array(p_opt),
        "r1_naive": np.array(r1_naive),
        "r2_naive": np.array(r2_naive),
        "r3_naive": np.array(r3_naive),
        "p_naive": np.array(p_naive),
        "r1_greedy": np.array(r1_greedy),
        "r2_greedy": np.array(r2_greedy),
        "r3_greedy": np.array(r3_greedy),
        "p_greedy": np.array(p_greedy),
    }

# ============================================================
# PLOTS
# ============================================================
def plot_3dart_4point_comparison(results3):
    sig = results3["sigma"]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("4-Punkte-Checkout mit 3 Darts", fontsize=16)

    axs[0].plot(sig, results3["p_opt"], 'o-', lw=2.5, label='Optimal')
    axs[0].plot(sig, results3["p_naive"], 's--', lw=2, label='Naiv')
    axs[0].plot(sig, results3["p_greedy"], '^-.', lw=2, label='Greedy')
    axs[0].set_ylabel("Checkout-Wahrscheinlichkeit")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(sig, results3["p_opt"] - results3["p_naive"], 'o-', lw=2, label='opt - naiv', color='purple')
    axs[1].plot(sig, results3["p_opt"] - results3["p_greedy"], 's--', lw=2, label='opt - greedy', color='green')
    axs[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"σ (mm)")
    axs[1].set_ylabel("Differenz")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Radienvergleich: 4-Punkte-Checkout mit 3 Darts (Optimum)", fontsize=16)
    ax.plot(sig, results3["r1_opt"], 'o-', lw=2.5, label='r1_opt (1. Dart)', color='darkblue')
    ax.plot(sig, results3["r2_opt"], '^-', lw=2.5, label='r2_opt (2. Dart)', color='darkred')
    ax.plot(sig, results3["r3_opt"], 'd-', lw=2.5, label='r3_opt (3. Dart)', color='darkgreen')
    ax.set_xlabel(r"σ (mm)")
    ax.set_ylabel("Radius r (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    resultsC2 = compute_resultsC2()
    result4 = compute_result4()
    results3 = build_3dart_results(result4, resultsC2)
    plot_3dart_4point_comparison(results3)
    print("\nAlle Plots erzeugt!")
  
