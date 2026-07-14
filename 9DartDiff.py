import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# BOARD + INTEGRAL + p_T20 + p_D1 (wie zuvor)
R_BULL_INNER = 6.35
R_BULL_OUTER = 15.9
R_TRIPLE_INNER = 99.0
R_TRIPLE_OUTER = 107.0
R_DOUBLE_INNER = 162.0
R_DOUBLE_OUTER = 170.0
segments = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5])
ANGLE_PER_SEG = 2 * np.pi / 20

def get_angle_for_segment_20():
    i = np.where(segments == 20)[0][0]
    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20

ANGLE_T20 = get_angle_for_segment_20()
THETA_MIN_T20 = ANGLE_T20 - ANGLE_PER_SEG / 2
THETA_MAX_T20 = ANGLE_T20 + ANGLE_PER_SEG / 2

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

def p_T20(r0, sigma):
    r0 = float(r0)
    return integrate_over_region(R_TRIPLE_INNER, R_TRIPLE_OUTER, THETA_MIN_T20, THETA_MAX_T20, r0, ANGLE_T20, sigma)

def p_D1(r0, sigma):
    r0 = float(r0)
    i = np.where(segments == 1)[0][0]
    angle_d = np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20
    theta_min_d = angle_d - ANGLE_PER_SEG / 2
    theta_max_d = angle_d + ANGLE_PER_SEG / 2
    return integrate_over_region(R_DOUBLE_INNER, R_DOUBLE_OUTER, theta_min_d, theta_max_d, r0, angle_d, sigma)

def find_max_pT20(sigma, bracket=(90, 130), tol=1e-6):
    def neg(r):
        return -p_T20(r, sigma)
    res = minimize_scalar(neg, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

def find_max_pD(sigma, bracket=(140, 190), tol=1e-6):
    def neg(r):
        return -p_D1(r, sigma)
    res = minimize_scalar(neg, bounds=bracket, tol=tol, method='bounded')
    return res.x, -res.fun

def nine_darter_E(rT, rD, sigma):
    pT = p_T20(rT, sigma)
    pD = p_D1(rD, sigma)
    return pT**8 * pD

def compare_nine_darter_log_diff():
    sigmas = np.linspace(5, 50, 20)
    E_naive = []
    E_opt = []
    rT_naive = 103.0
    rD_naive = 166.0
    for sigma in sigmas:
        sigma = float(sigma)
        E_n = nine_darter_E(rT_naive, rD_naive, sigma)
        rT_opt, _ = find_max_pT20(sigma)
        rD_opt, _ = find_max_pD(sigma)
        E_o = nine_darter_E(rT_opt, rD_opt, sigma)
        E_naive.append(E_n)
        E_opt.append(E_o)
    diff = np.array(E_opt) - np.array(E_naive)
    plt.figure(figsize=(12, 8))
    plt.semilogy(sigmas, diff, 'o-', lw=2.5, color='purple', label='Optimal - Naiv')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("σ (mm)")
    plt.ylabel("Differenz (log-Skala)")
    plt.title("Vorteil Optimal vs. Naiv – 9-Darter (logarithmisch)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_nine_darter_log_diff()
