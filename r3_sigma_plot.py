import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
from scipy.special import i0e


# -----------------------------
# Board-Parameter (mm)
# -----------------------------
R_BULL_INNER = 6.35
R_BULL_OUTER = 15.9
R_TRIPLE_INNER = 99
R_TRIPLE_OUTER = 107
R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170

segments = np.array([
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
])

ANGLE_PER_SEG = 2 * np.pi / 20

# ============================================================
# PLOT: r3*(sigma)
# kontinuierlich deterministisch optimiert
# ============================================================
# ------------------------------------------------------------

# ------------------------------------------------------------
# p_det
# ------------------------------------------------------------

def p_det(r, sigma):

    alpha = np.pi / 20

    def integrand_theta(theta, rho):

        return np.exp(
            -(rho**2 + r**2 - 2*rho*r*np.cos(theta))
            / (2*sigma**2)
        )

    def integrand_rho(rho):

        val, _ = quad(
            integrand_theta,
            -alpha,
            alpha,
            args=(rho,),
            epsabs=1e-8,
            epsrel=1e-8
        )

        return rho * val

    val, _ = quad(
        integrand_rho,
        R_DOUBLE_INNER,
        R_DOUBLE_OUTER,
        epsabs=1e-8,
        epsrel=1e-8
    )

    return val / (2 * np.pi * sigma**2)

# ------------------------------------------------------------
# q_det
# ------------------------------------------------------------

def Q_det(r, sigma):

    val, _ = quad(
        lambda rho: rice_pdf(rho, r, sigma),
        0,
        R_DOUBLE_OUTER,
        epsabs=1e-10,
        epsrel=1e-10
    )

    return val

def q_det(r, sigma):

    return Q_det(r, sigma) - p_det(r, sigma)

# ------------------------------------------------------------
# Optimales r3
# ------------------------------------------------------------

# Optimales r3
# ------------------------------------------------------------

def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded'
    )

    return res.x, p_det(res.x, sigma)

sigmas = np.linspace(2, 60, 40)

r3_vals = []
p3_vals = []

print("Berechne r3*(sigma)...")

for sigma in sigmas:

    r3_opt, p3_opt = find_r3_det(sigma)

    r3_vals.append(r3_opt)
    p3_vals.append(p3_opt)

    print(f"sigma={sigma:.2f} -> r3={r3_opt:.4f}")

r3_vals = np.array(r3_vals)
p3_vals = np.array(p3_vals)

# ------------------------------------------------
# Plot
# ------------------------------------------------
plt.figure(figsize=(8,5))

plt.plot(sigmas, r3_vals, linewidth=2)

plt.xlabel(r"$\sigma$ (mm)")
plt.ylabel(r"Optimaler Radius $r_3^\ast$ (mm)")
plt.title(r"Kontinuierlich optimierter Radius $r_3^\ast(\sigma)$")

plt.grid()
plt.tight_layout()
plt.show()
