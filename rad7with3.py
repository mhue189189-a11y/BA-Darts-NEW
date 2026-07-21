import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import i0e
from scipy.interpolate import interp1d

# ========================================================
# TEIL 1: r1_opt aus dem ersten Skript (C3-Optimierung)
# ========================================================
R_BULL_OUTER = 15.9
R_TRIPLE_INNER = 99
R_TRIPLE_OUTER = 107
R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170
ALPHA = np.pi / 20
R1_SMALL_SIGMA = 137.12

def gaussian_density(x, y, mu_x, mu_y, sigma):
    return (1 / (2 * np.pi * sigma**2) *
            np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)))

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
    return (rho / sigma**2 * np.exp(-((rho - r)**2) / (2 * sigma**2)) * i0e(x))

def Q_det(r, sigma):
    val, _ = quad(lambda rho: rice_pdf(rho, r, sigma), 0, R_DOUBLE_OUTER,
                  epsabs=1e-8, epsrel=1e-8, limit=200)
    return val

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

def p1_maximum(sigma):
    res = minimize_scalar(lambda r: -p1_det(r, sigma), bounds=(0, 200), method='bounded')
    return res.x

def greedy_r1(sigma):
    if sigma <= 10:
        return R1_SMALL_SIGMA
    return p1_maximum(sigma)

def C2_optimal(sigma):
    r2 = find_r3_det(sigma)
    p2 = p_det(r2, sigma)
    q2 = Q_det(r2, sigma) - p2
    p3 = p_det(r2, sigma)
    return p2 + (1 - p2 - q2) * p3

def C3_optimal(sigma):
    if sigma <= 10:
        r1 = R1_SMALL_SIGMA
    else:
        C2 = C2_optimal(sigma)
        r2 = find_r3_det(sigma)
        pD1 = p_det(r2, sigma)
        def objective(r):
            p1 = p1_det(r, sigma)
            p0 = p0_det(r, sigma)
            return -(p1 * C2 + p0 * (p1 * pD1))
        res = minimize_scalar(objective, bounds=(0, 200), method='bounded')
        r1 = res.x
    return r1

# ========================================================
# TEIL 2: r2 und r3 aus dem zweiten Skript
# ========================================================
R_BULL_INNER = 6.35
segments = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5])
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
    return integrate_over_region(R_DOUBLE_INNER, R_DOUBLE_OUTER, THETA_MIN_1, THETA_MAX_1, r0, ANGLE_1, sigma)

def p_single_1(r0, sigma):
    p_single = integrate_over_region(R_BULL_OUTER, R_DOUBLE_INNER, THETA_MIN_1, THETA_MAX_1, r0, ANGLE_1, sigma)
    p_triple = integrate_over_region(R_TRIPLE_INNER, R_TRIPLE_OUTER, THETA_MIN_1, THETA_MAX_1, r0, ANGLE_1, sigma)
    return p_single - p_triple

def p_miss(r0, sigma):
    def miss_int(rho):
        v, _ = quad(lambda th: rho * gaussian_polar_integrand(th, rho, r0, ANGLE_1, sigma), 0, 2*np.pi, epsabs=1e-6, epsrel=1e-6)
        return v
    val, _ = quad(miss_int, R_DOUBLE_OUTER, 1000, epsabs=1e-6, epsrel=1e-6)
    return val / (2 * np.pi * sigma**2)

def p_single_2(r0, sigma):
    p_single = integrate_over_region(R_BULL_OUTER, R_DOUBLE_INNER, THETA_MIN_2, THETA_MAX_2, r0, ANGLE_2, sigma)
    p_triple = integrate_over_region(R_TRIPLE_INNER, R_TRIPLE_OUTER, THETA_MIN_2, THETA_MAX_2, r0, ANGLE_2, sigma)
    return p_single - p_triple

def find_maximum_pD1(sigma, bracket=(140, 190)):
    def neg_pD1(r): return -p_D1(r, sigma)
    res = minimize_scalar(neg_pD1, bounds=bracket, method='bounded')
    return res.x

def find_maximum_E_4points_2darts(sigma, bracket=(120, 200)):
    r2_greedy = find_maximum_pD1(sigma, bracket=bracket)
    def neg_E(r1):
        p_d1 = p_D1(r1, sigma)
        p1 = p_single_1(r1, sigma)
        p0 = p_miss(r1, sigma)
        E = p_d1 + (p0 + p1) * p_D1(r2_greedy, sigma)
        return -E
    res = minimize_scalar(neg_E, bounds=bracket, method='bounded')
    return res.x, r2_greedy

# ========================================================
# Hauptberechnung & Plot
# ========================================================
sigmas = np.linspace(1, 60, 30)

r1_opts = []
r2_opts = []
r3_opts = []

for sigma in sigmas:
    r1 = C3_optimal(sigma)
    r1_opts.append(r1)
    
    _, r2 = find_maximum_E_4points_2darts(sigma)
    r2_opts.append(r2)
    r3_opts.append(r2)  # r3 = r2 in der aktuellen Logik

r1_opts = np.array(r1_opts)
r2_opts = np.array(r2_opts)
r3_opts = np.array(r3_opts)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(sigmas, r1_opts, 'o-', linewidth=2.5, label='r₁* (optimal)', color='darkblue')
plt.plot(sigmas, r2_opts, '^-', linewidth=2.5, label='r₂', color='darkred')
plt.plot(sigmas, r3_opts, 'd-', linewidth=2.5, label='r₃', color='darkgreen')

plt.xlabel(r'$\sigma$ (mm)')
plt.ylabel(r'Zielradius $r$ (mm)')
plt.title('Optimale Zielradien für den 7-Punkte-Checkout mit 3 Darts')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
