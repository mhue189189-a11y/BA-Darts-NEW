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
# KERN-FUNKTIONEN
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
def p_D1(r0, sigma):
    r0 = float(r0)
    return integrate_over_region(R_DOUBLE_INNER, R_DOUBLE_OUTER, THETA_MIN_1, THETA_MAX_1, r0, ANGLE_1, sigma)
def p_miss(r0, sigma):
    r0 = float(r0)
    def miss_int(rho):
        v, _ = quad(lambda th: rho * gaussian_polar_integrand(th, rho, r0, ANGLE_1, sigma), 0, 2*np.pi)
        return v
    val, _ = quad(miss_int, R_DOUBLE_OUTER, 1000, epsabs=1e-6)
    return val / (2 * np.pi * sigma**2)
# ============================================================
# OPTIMIERUNG
# ============================================================
def find_max_pD1(sigma, bracket=(130, 220), tol=1e-8):
    def neg(r): return -p_D1(r, sigma)
    res = minimize_scalar(neg, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun
def find_max_E_2darts(sigma, bracket=(130, 220), tol=1e-8):
    r_d1, p_d1 = find_max_pD1(sigma, bracket, tol)
    def neg_E(r):
        return -(p_D1(r, sigma) + p_miss(r, sigma) * p_d1)
    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun, r_d1, p_d1
def find_max_E_3darts(sigma, resultsC2, bracket=(130, 220), tol=1e-8):
    pE_2_opt = np.interp(sigma, resultsC2['sigma'], resultsC2['p_opt'])
    def neg_E(r):
        p_d1 = p_D1(r, sigma)
        p0 = p_miss(r, sigma)
        return -(p_d1 + p0 * pE_2_opt)
    res = minimize_scalar(neg_E, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun
# ============================================================
# COMMON SIGMAS
# ============================================================
COMMON_SIGMAS = np.linspace(1, 60, 30)
# ============================================================
# 2-DARTS CHECKOUT
# ============================================================
def checkout_comparison_2darts():
    r_opt = []
    r_naive = []
    r_greedy = []
    p_opt = []
    p_naive = []
    p_greedy = []
  
    print("=== 2-Darts 2-Punkte-Checkout ===\n")
  
    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        print(f"σ = {sigma:5.1f}")
      
        r1_opt, pE_opt, r_d1, _ = find_max_E_2darts(sigma)
      
        r_naiv = 166.0
        p_d1_naiv = p_D1(r_naiv, sigma)
        p0_naiv = p_miss(r_naiv, sigma)
        pE_naiv = p_d1_naiv + p0_naiv * p_d1_naiv
      
        p_d1_g = p_D1(r_d1, sigma)
        p0_g = p_miss(r_d1, sigma)
        pE_greedy = p_d1_g + p0_g * p_d1_g
      
        r_opt.append(r1_opt)
        r_naive.append(r_naiv)
        r_greedy.append(r_d1)
        p_opt.append(pE_opt)
        p_naive.append(pE_naiv)
        p_greedy.append(pE_greedy)
  
    resultsC2 = {
        'sigma': COMMON_SIGMAS.copy(),
        'r_opt': np.array(r_opt),
        'p_opt': np.array(p_opt),
        'r_naive': np.array(r_naive),
        'p_naive': np.array(p_naive),
        'r_greedy': np.array(r_greedy),
        'p_greedy': np.array(p_greedy)
    }
    return resultsC2
# ============================================================
# 3-DARTS CHECKOUT
# ============================================================
def checkout_comparison_3darts(resultsC2):
    r_opt_3 = []
    r_naive_3 = []
    r_greedy_3 = []
    p_opt_3 = []
    p_naive_3 = []
    p_greedy_3 = []
  
    print("\n=== 3-Darts 2-Punkte-Checkout ===\n")
  
    for sigma in COMMON_SIGMAS:
        sigma = float(sigma)
        print(f"σ = {sigma:5.1f}")
      
        r1_opt, pE_opt = find_max_E_3darts(sigma, resultsC2)
      
        r_naiv = 166.0
        p_d1_n = p_D1(r_naiv, sigma)
        p0_n = p_miss(r_naiv, sigma)
        pE2_n = np.interp(sigma, resultsC2['sigma'], resultsC2['p_opt'])
        pE_naiv = p_d1_n + p0_n * pE2_n
      
        r_d1_g, _ = find_max_pD1(sigma)
        p_d1_g = p_D1(r_d1_g, sigma)
        p0_g = p_miss(r_d1_g, sigma)
        pE2_g = np.interp(sigma, resultsC2['sigma'], resultsC2['p_opt'])
        pE_greedy = p_d1_g + p0_g * pE2_g
      
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
# UPDATED PLOTS
# ============================================================
def plot_updated_comparisons(resultsC2, resultsC3):
    sig = resultsC2['sigma']
   
    # Probability plots with differences
    # 2-Darts Strategy Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("2-Darts 2-Punkte-Checkout - Strategie-Vergleich", fontsize=16)
   
    # Probability
    axs[0].plot(sig, resultsC2['p_opt'], 'o-', lw=2.5, label='Optimal')
    axs[0].plot(sig, resultsC2['p_naive'], 's--', lw=2, label='Naiv')
    axs[0].plot(sig, resultsC2['p_greedy'], '^-.', lw=2, label='Greedy')
    axs[0].set_ylabel("Checkout-Wahrscheinlichkeit")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
   
    # Differences
    diff_opt_naiv_2 = resultsC2['p_opt'] - resultsC2['p_naive']
    diff_naiv_greedy_2 = resultsC2['p_naive'] - resultsC2['p_greedy']
    axs[1].plot(sig, diff_opt_naiv_2, 'o-', lw=2, label='opt - naiv', color='purple')
    axs[1].plot(sig, diff_naiv_greedy_2, 's--', lw=2, label='naiv - greedy', color='brown')
    axs[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"$\sigma$ (mm)")
    axs[1].set_ylabel("Differenz der Wahrscheinlichkeiten")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
   
    plt.tight_layout()
    plt.savefig("2darts_strategie_vergleich.png", dpi=300, bbox_inches='tight')
    plt.show()
   
    # 3-Darts Strategy Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("3-Darts 2-Punkte-Checkout - Strategie-Vergleich", fontsize=16)
   
    # Probability
    axs[0].plot(sig, resultsC3['p_opt_3'], 'o-', lw=2.5, label='Optimal', color='darkblue')
    axs[0].plot(sig, resultsC3['p_naive_3'], 's--', lw=2, label='Naiv', color='darkorange')
    axs[0].plot(sig, resultsC3['p_greedy_3'], '^-.', lw=2, label='Greedy', color='green')
    axs[0].set_ylabel("Checkout-Wahrscheinlichkeit")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
   
    # Differences
    diff_opt_naiv_3 = resultsC3['p_opt_3'] - resultsC3['p_naive_3']
    diff_naiv_greedy_3 = resultsC3['p_naive_3'] - resultsC3['p_greedy_3']
    axs[1].plot(sig, diff_opt_naiv_3, 'o-', lw=2, label='opt - naiv', color='purple')
    axs[1].plot(sig, diff_naiv_greedy_3, 's--', lw=2, label='naiv - greedy', color='brown')
    axs[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"$\sigma$ (mm)")
    axs[1].set_ylabel("Differenz der Wahrscheinlichkeiten")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
   
    plt.tight_layout()

    plt.savefig("3darts_strategie_vergleich.png", dpi=300, bbox_inches='tight')
    plt.show()
   
    # Combined Radius Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Vergleich der optimalen Zielradien für den 2 Punkte Checkout mit 3 Darts", fontsize=16)
   
    ax.plot(sig, resultsC3['r_opt_3'], 'o-', lw=2.5, label='r1_opt (3-Darts)', color='darkblue')
    ax.plot(sig, resultsC2['r_opt'], '^-', lw=2.5, label='r2_opt (2-Darts)', color='darkred')
    ax.plot(sig, resultsC2['r_naive'], 's--', lw=2, label='r_naiv = 166 mm', color='darkorange')
    ax.plot(sig, resultsC2['r_greedy'], 'd-.', lw=2, label='r3_opt = r_greedy', color='green')
   
    ax.set_xlabel(r"$\sigma$ (mm)")
    ax.set_ylabel("Aim-Radius r (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("radiusvergleich2mit3.png", dpi=300, bbox_inches='tight')
    plt.show()
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    resultsC2 = checkout_comparison_2darts()
    resultsC3_3 = checkout_comparison_3darts(resultsC2)
   
    plot_updated_comparisons(resultsC2, resultsC3)
   
    print("\nAlle Plots erzeugt!")
