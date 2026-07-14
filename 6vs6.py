import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
from scipy.special import i0e

# ============================================================
# GEMEINSAME BOARD PARAMETER & GEOMETRIE
# ============================================================
R_BULL_OUTER = 15.9
R_TRIPLE_INNER = 99.0
R_TRIPLE_OUTER = 107.0
R_DOUBLE_INNER = 162.0
R_DOUBLE_OUTER = 170.0
ALPHA = np.pi / 20

segments = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5])
ANGLE_PER_SEG = 2 * np.pi / 20

def get_angle_for_segment_1():
    i = np.where(segments == 1)[0][0]
    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20

ANGLE_1 = get_angle_for_segment_1()
THETA_MIN_1 = ANGLE_1 - ANGLE_PER_SEG / 2
THETA_MAX_1 = ANGLE_1 + ANGLE_PER_SEG / 2

# ============================================================
# MATHEMATISCHE FUNKTIONEN (AUS BEIDEN SKRIPTEN)
# ============================================================
def gaussian_density(x, y, mu_x, mu_y, sigma):
    return (1 / (2 * np.pi * sigma**2) * np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)))

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

# --- Wahrscheinlichkeiten für 6er-Weg (Skript 1) ---
def p_D1_s1(r0, sigma):
    return integrate_over_region(R_DOUBLE_INNER, R_DOUBLE_OUTER, THETA_MIN_1, THETA_MAX_1, float(r0), ANGLE_1, sigma)

def p_single_1_s1(r0, sigma):
    p_single = integrate_over_region(R_BULL_OUTER, R_DOUBLE_INNER, THETA_MIN_1, THETA_MAX_1, float(r0), ANGLE_1, sigma)
    p_triple = integrate_over_region(R_TRIPLE_INNER, R_TRIPLE_OUTER, THETA_MIN_1, THETA_MAX_1, float(r0), ANGLE_1, sigma)
    return p_single - p_triple

def p_miss_s1(r0, sigma):
    def miss_int(rho):
        v, _ = quad(lambda th: rho * gaussian_polar_integrand(th, rho, float(r0), ANGLE_1, sigma), 0, 2*np.pi, epsabs=1e-6, epsrel=1e-6)
        return v
    val, _ = quad(miss_int, R_DOUBLE_OUTER, 1000, epsabs=1e-6, epsrel=1e-6)
    return val / (2 * np.pi * sigma**2)

def find_maximum_pD1(sigma):
    res = minimize_scalar(lambda r: -p_D1_s1(r, sigma), bounds=(140, 190), method='bounded')
    return res.x, -res.fun

# --- Wahrscheinlichkeiten & Funktionen für 7er-Weg (Skript 2) ---
def p_det(r, sigma):
    def integrand(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return gaussian_density(x, y, r, 0, sigma) * rho
    val, _ = quad(lambda rho: quad(lambda theta: integrand(theta, rho), -ALPHA, ALPHA)[0],
                  R_DOUBLE_INNER, R_DOUBLE_OUTER, epsabs=1e-8, epsrel=1e-8)
    return val

def rice_pdf(rho, r, sigma):
    x = rho * r / sigma**2
    return (rho / sigma**2) * np.exp(-((rho - r)**2) / (2 * sigma**2)) * i0e(x)

def Q_det(r, sigma):
    val, _ = quad(lambda rho: rice_pdf(rho, r, sigma), 0, R_DOUBLE_OUTER, epsabs=1e-8, epsrel=1e-8, limit=200)
    return val

def q_det(r, sigma):
    return Q_det(r, sigma) - p_det(r, sigma)

def p1_det(r, sigma):
    intervals = [(R_BULL_OUTER, R_TRIPLE_INNER), (R_TRIPLE_OUTER, R_DOUBLE_INNER)]
    total = 0
    for r_min, r_max in intervals:
        def integrand(theta, rho):
            x = rho * np.cos(theta)
            y = rho * np.sin(theta)
            return gaussian_density(x, y, r, 0, sigma) * rho
        val, _ = quad(lambda rho: quad(lambda theta: integrand(theta, rho), -ALPHA, ALPHA)[0],
                      r_min, r_max, epsabs=1e-8, epsrel=1e-8)
        total += val
    return total

def p0_det(r, sigma):
    return 1 - Q_det(r, sigma)

def find_r3_det(sigma):
    res = minimize_scalar(lambda r: -p_det(r, sigma), bounds=(155, 175), method='bounded')
    return res.x

def checkout_value_det(r, rs, P_det_vals, Q_det_vals, r3):
    r1, r2 = r
    p1 = np.interp(r1, rs, P_det_vals)
    q1 = np.interp(r1, rs, Q_det_vals)
    p2 = np.interp(r2, rs, P_det_vals)
    q2 = np.interp(r2, rs, Q_det_vals)
    p3 = np.interp(r3, rs, P_det_vals)
    return -(p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3))

def optimize_r1_r2_det(rs, P_det_vals, Q_det_vals, r3):
    r_init = rs[np.argmax(P_det_vals)]
    res = minimize(checkout_value_det, x0=[r_init, r_init], args=(rs, P_det_vals, Q_det_vals, r3),
                   bounds=[(0, 250), (0, 250)], method='L-BFGS-B')
    return res.x[0], res.x[1], -res.fun

def optimal_strategy_3darts(sigma):
    rs = np.linspace(0, 250, 120)
    P_det_vals = np.array([p_det(r, sigma) for r in rs])
    Q_det_vals = np.array([q_det(r, sigma) for r in rs])
    r3 = find_r3_det(sigma)
    r1_opt, r2_opt, val_opt = optimize_r1_r2_det(rs, P_det_vals, Q_det_vals, r3)
    return r1_opt, r2_opt, r3, val_opt

def C2_optimal(sigma):
    _, r2_opt, r3_opt, _ = optimal_strategy_3darts(sigma)
    p2 = p_det(r2_opt, sigma)
    q2 = q_det(r2_opt, sigma)
    p3 = p_det(r3_opt, sigma)
    return p2 + (1 - p2 - q2) * p3

def C4_optimal(sigma):
    r_double_opt = find_r3_det(sigma)
    res = minimize_scalar(lambda r: -(p_det(r, sigma) + (p0_det(r, sigma) + p1_det(r, sigma)) * p_det(r_double_opt, sigma)),
                          bounds=(120, 200), method='bounded')
    return -res.fun

def C3_optimal(sigma):
    res = minimize_scalar(lambda r: -p1_det(r, sigma), bounds=(0, 200), method='bounded')
    r1_opt = res.x
    r2_opt = find_r3_det(sigma)
    return p1_det(r1_opt, sigma) * p_det(r2_opt, sigma)

def total_value(r, sigma, C2, C3):
    return p1_det(r, sigma) * C2 + p0_det(r, sigma) * C3

# ============================================================
# STRATEGIEBERECHNUNG ÜBER SIGMA
# ============================================================
def run_comparison():
    sigmas = np.linspace(3, 60, 25)
    
    # 1. Vorbereiten der C2 & C3 Interpolation für den 6er-Weg (Nicht Umstellen)
    c2_opt, c3_opt = [], []
    for sigma in sigmas:
        sigma_f = float(sigma)
        r_greedy, pD1_max = find_maximum_pD1(sigma_f)
        
        # C2
        res_c2 = minimize_scalar(lambda r: -(p_D1_s1(r, sigma_f) + p_miss_s1(r, sigma_f) * pD1_max),
                                 bounds=(130, 200), method='bounded')
        c2_opt.append(-res_c2.fun)
        # C3
        res_c3 = minimize_scalar(lambda r: -(p_single_1_s1(r, sigma_f) * c2_opt[-1]),
                                 bounds=(110, 170), method='bounded')
        c3_opt.append(-res_c3.fun)

    get_c2_opt = interp1d(sigmas, c2_opt, kind='linear', fill_value="extrapolate")
    get_c3_opt = interp1d(sigmas, c3_opt, kind='linear', fill_value="extrapolate")

    vals_nicht_umstellen = [] # 6er-Optimal-Weg
    vals_umstellen = []       # 7er-Optimal-Weg (aus Skript 2)

    for sigma in sigmas:
        sigma_f = float(sigma)
        print(f"Berechne σ = {sigma_f:.1f}...")

        # --- A. Nicht Umstellen (6 Punkte Checkout Optimal) ---
        c2_val = float(get_c2_opt(sigma_f))
        c3_val = float(get_c3_opt(sigma_f))
        
        res_6 = minimize_scalar(lambda r1: -(p_D1_s1(r1, sigma_f) + p_single_1_s1(r1, sigma_f) * c3_val + p_miss_s1(r1, sigma_f) * c2_val),
                                bounds=(110, 200), method='bounded')
        vals_nicht_umstellen.append(-res_6.fun)

        # --- B. Umstellen (Optimaler 7er-Weg aus Skript 2) ---
        rs = np.linspace(0, 200, 80)
        C2_opt = C2_optimal(sigma_f)
        C3_opt_val = C3_optimal(sigma_f)
        C4_opt_val = C4_optimal(sigma_f)
        
        q_opt = C4_opt_val / C2_opt
        total_opt_vals = np.array([total_value(r, sigma_f, C2_opt, C3_opt_val) for r in rs]) * q_opt
        vals_umstellen.append(np.max(total_opt_vals))

    return sigmas, np.array(vals_nicht_umstellen), np.array(vals_umstellen)

# ============================================================
# PLOTTEN DER ERGEBNISSE
# ============================================================
def plot_results(sigmas, p_no_switch, p_switch):
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Plotten der exakten Kurven
    ax.plot(sigmas, p_no_switch, 'o-', linewidth=2.5, color='darkblue', label='Nicht Umstellen')
    ax.plot(sigmas, p_switch, 's--', linewidth=2.5, color='darkorange', label='Umstellen')
    
    ax.set_xlabel(r"$\sigma$ (mm)", fontsize=12)
    ax.set_ylabel("Erfolgswahrscheinlichkeit", fontsize=12)
    ax.set_title("6 Punkte Checkout mit 3 Darts Vergleich Umstellen vs Nicht Umstellen", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sigmas, p_no_switch, p_switch = run_comparison()
    plot_results(sigmas, p_no_switch, p_switch)
