import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0e
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar

# -----------------------------
# Board
# -----------------------------
R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170

# -----------------------------
# Rice PDF
# -----------------------------
def rice_pdf(rho, r, sigma):
    x = rho * r / sigma**2
    return (rho / sigma**2) * \
           np.exp(-((rho - r)**2) / (2*sigma**2)) * \
           i0e(x)

# -----------------------------
# p_det
# -----------------------------
def p_det(r, sigma):

    alpha = np.pi / 20

    def integrand_theta(theta, rho):
        return np.exp(
            -(rho**2 + r**2 - 2*rho*r*np.cos(theta)) / (2*sigma**2)
        )

    def integrand_rho(rho):
        val, _ = quad(
            integrand_theta,
            -alpha, alpha,
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

# -----------------------------
# q_det
# -----------------------------
def Q_det(r, sigma):
    val, _ = quad(
        lambda rho: rice_pdf(rho, r, sigma),
        0, R_DOUBLE_OUTER,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return val

def q_det(r, sigma):
    return Q_det(r, sigma) - p_det(r, sigma)

# -----------------------------
# r3 optimal
# -----------------------------
def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded'
    )

    return res.x, p_det(res.x, sigma)

# -----------------------------
# Checkout Wert (3 Darts)
# -----------------------------
def checkout_value_fixed_strategy(r1, r2, r3, sigma):

    p1 = p_det(r1, sigma)
    q1 = q_det(r1, sigma)

    p2 = p_det(r2, sigma)
    q2 = q_det(r2, sigma)

    p3 = p_det(r3, sigma)

    return p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3)

# -----------------------------
# Split nach Dart
# -----------------------------
def checkout_split_3darts(r1, r2, r3, sigma):

    p1 = p_det(r1, sigma)
    q1 = q_det(r1, sigma)

    p2 = p_det(r2, sigma)
    q2 = q_det(r2, sigma)

    p3 = p_det(r3, sigma)

    P1 = p1
    P2 = (1 - p1 - q1) * p2
    P3 = (1 - p1 - q1) * (1 - p2 - q2) * p3

    return P1, P2, P3

# -----------------------------
# Optimierung r1, r2
# -----------------------------
def checkout_value_det(r, rs, P_det_vals, Q_det_vals, r3):

    r1, r2 = r

    p1 = np.interp(r1, rs, P_det_vals)
    q1 = np.interp(r1, rs, Q_det_vals)

    p2 = np.interp(r2, rs, P_det_vals)
    q2 = np.interp(r2, rs, Q_det_vals)

    p3 = np.interp(r3, rs, P_det_vals)

    val = p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3)

    return -val

def optimize_r1_r2_det(rs, P_det_vals, Q_det_vals, r3):

    r_init = rs[np.argmax(P_det_vals)]

    res = minimize(
        checkout_value_det,
        x0=[r_init, r_init],
        args=(rs, P_det_vals, Q_det_vals, r3),
        bounds=[(0, 250), (0, 250)],
        method='L-BFGS-B'
    )

    return res.x[0], res.x[1], -res.fun

# -----------------------------
# Optimale Strategie
# -----------------------------
def optimal_strategy_det(sigma):

    rs = np.linspace(0, 250, 120)

    P_det_vals = np.array([p_det(r, sigma) for r in rs])
    Q_det_vals = np.array([q_det(r, sigma) for r in rs])

    r3, _ = find_r3_det(sigma)

    r1_opt, r2_opt, val_opt = optimize_r1_r2_det(
        rs, P_det_vals, Q_det_vals, r3
    )

    return r1_opt, r2_opt, r3, val_opt

def checkout_efficiency_per_dart(r1, r2, r3, sigma):

    # Dart 1
    p1 = p_det(r1, sigma)
    q1 = q_det(r1, sigma)

    # Dart 2
    p2 = p_det(r2, sigma)
    q2 = q_det(r2, sigma)

    # Dart 3
    p3 = p_det(r3, sigma)

    # -----------------------------
    # Checkout-Wkeiten
    # -----------------------------
    P1 = p1

    P2 = (1 - p1 - q1) * p2

    P3 = (1 - p1 - q1) * (1 - p2 - q2) * p3

    P_total = P1 + P2 + P3

    # -----------------------------
    # Erwartete Dartanzahl
    # -----------------------------
    E_darts = (
        1
        + (1 - p1 - q1)
        + (1 - p1 - q1) * (1 - p2 - q2)
    )

    # -----------------------------
    # Checkoutquote pro Dart
    # -----------------------------
    eta = P_total / E_darts

    return eta, P1, P2, P3, E_darts

# -----------------------------
# Hauptvergleich
# -----------------------------
def compare_strategies_vs_sigma():

    sigmas = np.linspace(1, 80, 50)

    vals_opt, vals_naive, vals_greedy = [], [], []

    P1_list, P2_list, P3_list = [], [], []

    p1_vals, p2_vals, p3_vals = [], [], []
    etas_opt = []
    etas_naive = []
    etas_greedy = []   
    
    for sigma in sigmas:

        print(f"σ = {sigma:.1f}")

        # Optimal
        r1_opt, r2_opt, r3_opt, val_opt = optimal_strategy_det(sigma)
        
        vals_opt.append(val_opt)
        p1_vals.append(p_det(r1_opt, sigma))
        p2_vals.append(p_det(r2_opt, sigma))
        p3_vals.append(p_det(r3_opt, sigma))
        eta_opt, _, _, _, _ = checkout_efficiency_per_dart(
            r1_opt,
            r2_opt,
            r3_opt,
            sigma
        )

        etas_opt.append(eta_opt)
        

        # Split
        P1, P2, P3 = checkout_split_3darts(
            r1_opt, r2_opt, r3_opt, sigma
        )
        P1_list.append(P1)
        P2_list.append(P2)
        P3_list.append(P3)

        # Naiv
        r_naive = 166
        vals_naive.append(
            checkout_value_fixed_strategy(r_naive, r_naive, r_naive, sigma)
        )
        eta_naive, _, _, _, _ = checkout_efficiency_per_dart(
            r_naive,
            r_naive,
            r_naive,
            sigma
        )
        
        etas_naive.append(eta_naive)

        # Greedy
        r3_det, _ = find_r3_det(sigma)
        vals_greedy.append(
            checkout_value_fixed_strategy(r3_det, r3_det, r3_det, sigma)
        )

        eta_greedy, _, _, _, _ = checkout_efficiency_per_dart(
            r3_det,
            r3_det,
            r3_det,
            sigma
        )

        etas_greedy.append(eta_greedy)

    #Chekout-Quoten
    etas_opt = np.array(etas_opt)
    etas_naive = np.array(etas_naive)
    etas_greedy = np.array(etas_greedy)
    
    p1_vals = np.array(p1_vals)
    p2_vals = np.array(p2_vals)
    p3_vals = np.array(p3_vals)
    # Arrays
    vals_opt = np.array(vals_opt)
    vals_naive = np.array(vals_naive)
    vals_greedy = np.array(vals_greedy)

    P1_list = np.array(P1_list)
    P2_list = np.array(P2_list)
    P3_list = np.array(P3_list)

    # -----------------------------
    # Plot 1: Strategien
    # -----------------------------
    plt.figure(figsize=(9,6))

    plt.plot(sigmas, vals_opt, label="Optimal", linewidth=2)
    plt.plot(sigmas, vals_naive, '--', label="Naiv")
    plt.plot(sigmas, vals_greedy, ':', label="Greedy")
    plt.plot(sigmas, vals_opt/3, label="Optimale Checkout-Quote", linewidth=2)

    plt.plot(sigmas, vals_opt - vals_naive, label="Opt - Naiv")
    plt.plot(sigmas, vals_opt - vals_greedy, label="Opt - Greedy")

    plt.xlabel("σ (mm)")
    plt.ylabel("Checkout-Wahrscheinlichkeit")
    plt.title("3 Darts: Strategievergleich")

    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------
    # Plot 2: Wann wird gecheckt
    # -----------------------------
    plt.figure(figsize=(9,6))

    plt.plot(sigmas, P1_list, label="Dart 1")
    plt.plot(sigmas, P2_list, label="Dart 2")
    plt.plot(sigmas, P3_list, label="Dart 3")

    plt.plot(sigmas, P1_list + P2_list + P3_list,
             '--', label="Summe", alpha=0.5)

    plt.xlabel("σ (mm)")
    plt.ylabel("Wahrscheinlichkeit")
    plt.title("Checkout-Verteilung (optimal)")

    plt.legend()
    plt.grid()
    plt.show()

    #Plot 3

    plt.figure(figsize=(9,6))

    plt.plot(sigmas, p1_vals, label="p(r1*)")
    plt.plot(sigmas, p2_vals, label="p(r2*)")
    plt.plot(sigmas, p3_vals, label="p(r3*)")
    
    plt.xlabel("σ (mm)")
    plt.ylabel("Trefferwahrscheinlichkeit p(r)")
    plt.title("Gewählte Ziel-Wahrscheinlichkeiten (optimale Strategie)")
    
    plt.legend()
    plt.grid()
    plt.show()
    #Plot 4

    plt.figure(figsize=(9,6))
    
    plt.plot(sigmas, etas_opt, linewidth=2,
             label="Optimal")
    
    plt.plot(sigmas, etas_naive, '--',
             label="Naiv")
    
    plt.plot(sigmas, etas_greedy, ':',
             label="Greedy")
    
    # Differenzen
   
    
    plt.plot(sigmas,
             -(etas_opt - etas_greedy),
             label="Diff Greedy - Opt",
             alpha=0.8)
    
    plt.xlabel("σ (mm)")
    plt.ylabel("Mittlere Checkoutquote pro Dart")
    
    plt.title("Checkout-Effizienz pro Dart")
    
    plt.legend()
    plt.grid()
    
    plt.show()
# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    compare_strategies_vs_sigma()
