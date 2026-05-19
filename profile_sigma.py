import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Board
# -----------------------------
R_BULL_INNER = 6.35
R_BULL_OUTER = 15.9
R_TRIPLE_INNER = 99
R_TRIPLE_OUTER = 107
R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170

segments = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                     3, 19, 7, 16, 8, 11, 14, 9, 12, 5])

ANGLE_PER_SEG = 2 * np.pi / 20


# -----------------------------
# Winkel Mitte Segment 1
# -----------------------------
def get_angle_segment1():
    i = 1  # Segment 1 Index
    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20

def get_angle_for_segment(seg_value):
    i = np.where(segments == seg_value)[0][0]
    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20


# -----------------------------
# p und q
# -----------------------------
def compute_p_q(center, sigma, samples=150000):

    x = np.random.normal(center[0], sigma, samples)
    y = np.random.normal(center[1], sigma, samples)

    r = np.sqrt(x**2 + y**2)

    theta = np.arctan2(y, x)
    theta = (np.pi/2 - theta + np.pi/20) % (2*np.pi)

    seg_index = np.floor(theta / ANGLE_PER_SEG).astype(int)
    seg_index = np.clip(seg_index, 0, 19)
    base = segments[seg_index]

    hit_d1 = (
        (r >= R_DOUBLE_INNER) & (r <= R_DOUBLE_OUTER) &
        (base == 1)
    )

    scores = np.zeros_like(r)

    scores[r <= R_BULL_INNER] = 50
    mask_outer = (r > R_BULL_INNER) & (r <= R_BULL_OUTER)
    scores[mask_outer] = 25

    mask_triple = (r >= R_TRIPLE_INNER) & (r <= R_TRIPLE_OUTER)
    scores[mask_triple] = 3 * base[mask_triple]

    mask_double = (r >= R_DOUBLE_INNER) & (r <= R_DOUBLE_OUTER)
    scores[mask_double] = 2 * base[mask_double]

    mask_single = (r > R_BULL_OUTER) & (r < R_DOUBLE_INNER)
    scores[mask_single & ~mask_triple & ~mask_double] = base[mask_single & ~mask_triple & ~mask_double]

    bust = (scores > 0) | ((scores == 2) & (~hit_d1))

    return np.mean(hit_d1), np.mean(bust)


# -----------------------------
# p(r), q(r) entlang Linie
# -----------------------------
def compute_radial_profile(sigma):

    angle = get_angle_segment1()

    rs = np.linspace(0, R_DOUBLE_OUTER + 80, 120)

    P = []
    Q = []

    print("Berechne p(r), q(r)...")

    for r in rs:
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        p, q = compute_p_q((x, y), sigma)
        P.append(p)
        Q.append(q)

    return rs, np.array(P), np.array(Q)


def compute_profile_for_segment(sigma, seg_value):

    angle = get_angle_for_segment(seg_value)

    rs = np.linspace(0, R_DOUBLE_OUTER + 80, 120)

    Q = []

    print(f"Berechne q(r) für Segment {seg_value}...")

    for r in rs:
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        _, q = compute_p_q((x, y), sigma)
        Q.append(q)

    return rs, np.array(Q)

# -----------------------------
# Optimierung (nur 1D!)
# -----------------------------
def optimize_checkout(rs, P, Q):

    best = None
    best_val = -1

    n = len(rs)

    for i in range(n):
        for j in range(n):
            for k in range(n):

                p1, q1 = P[i], Q[i]
                p2, q2 = P[j], Q[j]
                p3, q3 = P[k], Q[k]

                val = p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3)

                if val > best_val:
                    best_val = val
                    best = (rs[i], rs[j], rs[k])

    return best, best_val
# -----------------------------
# Heatmap r1, r2
# -----------------------------
def plot_r1_r2_heatmap(rs, P, Q):

    # 🔥 r3 fixieren
    idx3 = np.argmax(P)
    r3 = rs[idx3]
    p3 = P[idx3]

    print(f"Fixiertes r3 = {r3:.2f}")

    n = len(rs)
    Z = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            p1, q1 = P[i], Q[i]
            p2, q2 = P[j], Q[j]

            val = p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3)

            Z[j, i] = val

    # Plot
    idx = np.unravel_index(np.argmax(Z), Z.shape)
    i_max, j_max = idx[1], idx[0]
    
    r1_max = rs[i_max]
    r2_max = rs[j_max]
    z_max = Z[j_max, i_max]
    
    print(f"Maximum bei r1 = {r1_max:.2f}, r2 = {r2_max:.2f}")
    print(f"Max Checkout-Wkeit = {z_max:.4f}")
    
    # Plot
    plt.figure(figsize=(7,6))
    plt.imshow(Z, extent=[rs[0], rs[-1], rs[0], rs[-1]],
               origin='lower', aspect='auto')
    
    plt.colorbar(label="Checkout-Wahrscheinlichkeit")
    
    # 🔴 Maximum einzeichnen
    plt.scatter(r1_max, r2_max, marker='x', s=100)
    plt.contour(rs, rs, Z, levels=10, linewidths=0.8)
    
    plt.xlabel("r1")
    plt.ylabel("r2")
    plt.title("Heatmap für r1, r2 (r3 fixiert)")
    
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0e
from scipy.integrate import quad

# -----------------------------
# Deterministische Rice-Dichte
# -----------------------------
def rice_pdf(rho, r, sigma):
    x = rho * r / sigma**2
    return (rho / sigma**2) * \
           np.exp(-((rho - r)**2) / (2*sigma**2)) * \
           i0e(x)


# -----------------------------
# Deterministisches p(r)
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
# Deterministisches Q(r)
# (alles auf dem Board)
# -----------------------------
def Q_det(r, sigma):
    val, _ = quad(
        lambda rho: rice_pdf(rho, r, sigma),
        0, R_DOUBLE_OUTER,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return val


# -----------------------------
# Deterministisches q(r)
# -----------------------------
def q_det(r, sigma):
    return Q_det(r, sigma) - p_det(r, sigma)


# -----------------------------
# Vergleichsplot
# -----------------------------
def compare_models(sigma):

    angle = get_angle_segment1()

    rs = np.linspace(0, 250, 60)

    P_mc, Q_mc = [], []
    P_det, Q_det_list = [], []

    print("Vergleiche Monte Carlo vs deterministisch...")

    for r in rs:

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Monte Carlo
        p_mc, q_mc = compute_p_q((x, y), sigma, samples=80000)

        # Deterministisch
        p_d = p_det(r, sigma)
        q_d = q_det(r, sigma)

        P_mc.append(p_mc)
        Q_mc.append(q_mc)

        P_det.append(p_d)
        Q_det_list.append(q_d)

    P_mc = np.array(P_mc)
    Q_mc = np.array(Q_mc)

    P_det = np.array(P_det)
    Q_det_list = np.array(Q_det_list)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(9,5))

    # Checkout
    plt.plot(rs, P_mc, 'o', label="p(r) Monte Carlo", alpha=0.5)
    plt.plot(rs, P_det, '-', label="p(r) deterministisch")

    # Bust
    plt.plot(rs, Q_mc, 'o', label="q(r) Monte Carlo", alpha=0.5)
    plt.plot(rs, Q_det_list, '-', label="q(r) deterministisch")

    plt.xlabel("Radius r (mm)")
    plt.ylabel("Wahrscheinlichkeit")
    plt.title(f"Monte Carlo vs deterministisch (σ={sigma})")
    plt.legend()
    plt.grid()

    plt.show()

    # -----------------------------
    # Maxima vergleichen
    # -----------------------------
    r_mc = rs[np.argmax(P_mc)]
    r_det = rs[np.argmax(P_det)]

    print("\n===== Vergleich =====")
    print(f"MC Maximum p(r) bei r ≈ {r_mc:.2f}")
    print(f"Det Maximum p(r) bei r ≈ {r_det:.2f}")


from scipy.optimize import minimize_scalar

def find_r3_det(sigma):

    res = minimize_scalar(
        lambda r: -p_det(r, sigma),
        bounds=(155, 175),
        method='bounded',
        options={'xatol': 1e-5}
    )

    return res.x, p_det(res.x, sigma)

def p_mc(r, sigma, samples=200000):

    angle = get_angle_segment1()
    x = r * np.cos(angle)
    y = r * np.sin(angle)

    p, _ = compute_p_q((x, y), sigma, samples=samples)
    return p

def find_r3_mc(sigma):

    res = minimize_scalar(
        lambda r: -p_mc(r, sigma, samples=200000),
        bounds=(155, 175),
        method='bounded',
        options={'xatol': 0.1}
    )

    return res.x, p_mc(res.x, sigma, samples=400000)

def optimize_checkout_fixed_r3(rs, P, Q, r3, P_ref, Q_ref):

    # p3/q3 aus Referenz (z.B. deterministisch)
    p3 = np.interp(r3, rs, P_ref)
    q3 = np.interp(r3, rs, Q_ref)

    best = None
    best_val = -1

    n = len(rs)

    for i in range(n):
        for j in range(n):

            p1, q1 = P[i], Q[i]
            p2, q2 = P[j], Q[j]

            val = p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3)

            if val > best_val:
                best_val = val
                best = (rs[i], rs[j], r3)

    return best, best_val
    
def plot_r1_r2_heatmap_det(rs, P_det, Q_det, r3):

    p3 = np.interp(r3, rs, P_det)
    q3 = np.interp(r3, rs, Q_det)
    
    print(f"[DET] Fixiertes r3 = {r3:.4f}")

    n = len(rs)
    Z = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            p1, q1 = P_det[i], Q_det[i]
            p2, q2 = P_det[j], Q_det[j]

            val = p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3)

            Z[j, i] = val

    # -----------------------------
    # Maximum
    # -----------------------------
    idx = np.unravel_index(np.argmax(Z), Z.shape)

    r1_max = rs[idx[1]]
    r2_max = rs[idx[0]]
    z_max = Z[idx]

    print(f"[DET] Maximum bei r1 = {r1_max:.2f}, r2 = {r2_max:.2f}")
    print(f"[DET] Checkout-Wkeit = {z_max:.6f}")

    # -----------------------------
    # 🔥 Geometrisches Zentrum der Top-Kontur
    # -----------------------------
    threshold = z_max * 0.995   # z.B. 99.5% Niveau

    mask = Z >= threshold

    r1_grid, r2_grid = np.meshgrid(rs, rs)

    r1_center = np.mean(r1_grid[mask])
    r2_center = np.mean(r2_grid[mask])

    # p am Zentrum berechnen
    p1_c = np.interp(r1_center, rs, P_det)
    q1_c = np.interp(r1_center, rs, Q_det)

    p2_c = np.interp(r2_center, rs, P_det)
    q2_c = np.interp(r2_center, rs, Q_det)

    p_center = p1_c + (1 - p1_c - q1_c) * (p2_c + (1 - p2_c - q2_c) * p3)
    # Exaktes r3
    r3_det, _ = find_r3_det(sigma)
    
    # Kontinuierliche Optimierung
    r1_opt, r2_opt, val_opt = optimize_r1_r2_det(
        rs,
        P_det_vals,
        Q_det_vals,
        r3_det
    )

    print("\n[DET] Geometrisches Zentrum der Top-Kontur:")
    print(f"r1 = {r1_center:.4f}")
    print(f"r2 = {r2_center:.4f}")
    print(f"p  = {p_center:.6f}")

    print(f"Dr1 = {r1_opt-r1_center}")
    print(f"Dr2 = {r2_opt-r2_center}")


    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(7,6))

    plt.imshow(Z,
               extent=[rs[0], rs[-1], rs[0], rs[-1]],
               origin='lower',
               aspect='auto')

    plt.colorbar(label="Checkout-Wahrscheinlichkeit")
    #Konturen
    z_max = np.max(Z)
    z_min = np.min(Z)
    
    # 🔥 feine Levels nahe Maximum
    levels_fine = np.linspace(z_max - 0.002, z_max, 15)
    
    # 🔥 grobe Levels für Rest
    levels_coarse = np.linspace(z_min, z_max - 0.002, 8)
    
    # 🔥 kombinieren + sortieren + doppelte entfernen
    levels = np.unique(np.concatenate([levels_coarse, levels_fine]))
    plt.contour(rs, rs, Z, levels=10, linewidths=0.8)
    '''
    plt.contour(
    rs, rs, Z,
    levels=levels,
    colors='black',        # 🔥 hoher Kontrast
    linewidths=0.2        # 🔥 dicker
     )
     '''

    # Maximum
    plt.scatter(r1_max, r2_max, marker='x', s=50, label="Maximum")

    # 🔥 Zentrum
    plt.scatter(r1_center, r2_center, marker='x', s=50, label="Kontur-Zentrum")
    #Kontinuierlich
    plt.scatter(r1_opt, r2_opt, marker='x', s=50, label="Kontin. Max.")

    plt.xlabel("r1")
    plt.ylabel("r2")
    plt.title("DET: Heatmap + Kontur-Zentrum")

    plt.legend()
    plt.show()

def compare_heatmaps(rs, P_mc, Q_mc, P_det, Q_det):

    idx3_mc = np.argmax(P_mc)
    idx3_det = np.argmax(P_det)

    p3_mc = P_mc[idx3_mc]
    p3_det = P_det[idx3_det]

    n = len(rs)

    Z_mc = np.zeros((n, n))
    Z_det = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            p1_mc, q1_mc = P_mc[i], Q_mc[i]
            p2_mc, q2_mc = P_mc[j], Q_mc[j]

            p1_det, q1_det = P_det[i], Q_det[i]
            p2_det, q2_det = P_det[j], Q_det[j]

            Z_mc[j, i] = p1_mc + (1 - p1_mc - q1_mc) * (p2_mc + (1 - p2_mc - q2_mc) * p3_mc)
            Z_det[j, i] = p1_det + (1 - p1_det - q1_det) * (p2_det + (1 - p2_det - q2_det) * p3_det)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(Z_mc, extent=[rs[0], rs[-1], rs[0], rs[-1]], origin='lower')
    plt.title("MC Heatmap")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(Z_det, extent=[rs[0], rs[-1], rs[0], rs[-1]], origin='lower')
    plt.title("DET Heatmap")
    plt.colorbar()

    plt.show()

from scipy.optimize import minimize

def checkout_value_det(r, rs, P_det, Q_det, r3):

    r1, r2 = r

    # Interpolation (entscheidend!)
    p1 = np.interp(r1, rs, P_det)
    q1 = np.interp(r1, rs, Q_det)

    p2 = np.interp(r2, rs, P_det)
    q2 = np.interp(r2, rs, Q_det)

    p3 = np.interp(r3, rs, P_det)

    val = p1 + (1 - p1 - q1) * (p2 + (1 - p2 - q2) * p3)

    return -val  # Minimizer → minus
    
def optimize_r1_r2_det_continuous(sigma, r3):

    res = minimize(
        checkout_value_det_continuous,
        x0=[190, 180],   # guter Startwert
        args=(sigma, r3),
        bounds=[(0, 250), (0, 250)],
        method='L-BFGS-B'
    )

    r1_opt, r2_opt = res.x
    val_opt = -res.fun

    return r1_opt, r2_opt, val_opt

def optimize_r1_r2_det(rs, P_det, Q_det, r3):

    # Startwert = dein bisheriges Grid-Maximum
    r1_init = rs[np.argmax(P_det)]
    r2_init = r1_init

    res = minimize(
        checkout_value_det,
        x0=[r1_init, r2_init],
        args=(rs, P_det, Q_det, r3),
        bounds=[(0, 250), (0, 250)],
        method='L-BFGS-B'
    )

    r1_opt, r2_opt = res.x
    val_opt = -res.fun

    return r1_opt, r2_opt, val_opt

# ============================================================
# 2-DART CHECKOUT AUF D1
# ============================================================

# ============================================================
# 2-DART CHECKOUT
# letzter Dart FIX = greedy Radius
# ============================================================

def checkout_value_2darts(r1, rs, P_det, Q_det, r_last):

    p1 = np.interp(r1, rs, P_det)
    q1 = np.interp(r1, rs, Q_det)

    p_last = p_det(r_last, sigma)

    val = p1 + (1 - p1 - q1) * p_last

    return -val


# ============================================================
# 2-DART OPTIMIERUNG (KORREKT)
# ============================================================

def optimize_2darts_det(rs, P_det, Q_det, sigma):

    # letzter Dart = greedy
    r_last, p_last = find_r3_det(sigma)

    # Erwartungswertfunktion
    def value(r1):

        p1 = np.interp(r1, rs, P_det)
        q1 = np.interp(r1, rs, Q_det)

        return p1 + (1 - p1 - q1) * p_last

    # feines Grid statt unstable scalar minimize
    r_grid = np.linspace(0, 250, 5000)

    vals = np.array([value(r) for r in r_grid])

    idx = np.argmax(vals)

    r1_opt = r_grid[idx]
    val_opt = vals[idx]

    return r1_opt, r_last, val_opt
# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    sigma = 10

    rs, P, q = compute_radial_profile(sigma)
    Q = q-P
    # Plot
    plt.close()
    plt.figure(figsize=(8,5))
    P_det_vals = []
    Q_det_vals = []
    
    for r in rs:
        P_det_vals.append(p_det(r, sigma))
        Q_det_vals.append(q_det(r, sigma))
    
    P_det_vals = np.array(P_det_vals)
    Q_det_vals = np.array(Q_det_vals)
    #plt.plot(rs, P, label="p(r) = Check",linewidth=0.5)
    #plt.plot(rs, Q, label="q(r) = Bust",linewidth=0.5)
    #plt.plot(rs, q, label="Q(r) = Bust oder Check")
    plt.plot(rs, P_det_vals, label="p_det",linewidth=0.5)
    plt.plot(rs, Q_det_vals, label="q_det",linewidth=0.5)


    
    
    plt.xlabel("Radius (mm)")
    plt.ylabel("Wahrscheinlichkeit")
    plt.title(f"Profil entlang Segment 1 (σ={sigma})")
    plt.grid()

    

    # Optimierung
    best_r, best_val = optimize_checkout(rs, P, Q)

    #plt.vlines(best_r[0], 0, 1, label=f"Dart 1: r = {best_r[0]:.1f}", color="black")  
    #plt.vlines(best_r[1], 0, 1, label=f"Dart 2: r = {best_r[1]:.1f}", color="black")    
    plt.vlines(best_r[2], 0, 1, label=f"Dart 3_MC: r = r_det +/- 0.5", color="red", linewidth=0.5)    



    print("\nOptimale Radien:")
    print(f"Dart 1: r = {best_r[0]:.1f}")
    print(f"Dart 2: r = {best_r[1]:.1f}")
    print(f"Dart 3: r = {best_r[2]:.1f}")
    print(f"Checkout-Wahrscheinlichkeit: {best_val:.4f}")

    #plot_r1_r2_heatmap(rs, P, Q)

    r_det, p_det_val = find_r3_det(sigma)
    r_mc, p_mc_val = find_r3_mc(sigma)
    
    print("\n===== r3 Vergleich =====")
    print(f"Deterministisch: r = {r_det:.4f}, p = {p_det_val:.6f}")
    print(f"Monte Carlo   : r = {r_mc:.2f}, p = {p_mc_val:.6f}")

    
    plt.vlines(r_det, 0, 1, label=f"Dart 3_det: r = {r_det:.2f}", color="black", linewidth=0.5)  

    plt.legend()
    plt.show()

        # Deterministische Profile
    P_det_vals = np.array([p_det(r, sigma) for r in rs])
    Q_det_vals = np.array([q_det(r, sigma) for r in rs])
    
    # -----------------------------
    # Optimierung vergleichen
    # -----------------------------
     # 🔥 Exaktes r3 (DET!)
    r3_det, _ = find_r3_det(sigma)
    
    # DET (konsistent)
    best_det, val_det = optimize_checkout_fixed_r3(
        rs,
        P_det_vals,
        Q_det_vals,
        r3_det,
        P_det_vals,
        Q_det_vals
    )
    
    # MC (aber mit Grid r3!)
    best_mc, val_mc = optimize_checkout_fixed_r3(
        rs,
        P,
        Q,
        best_r[2],          # 🔥 wichtig!
        P_det_vals,      # 🔥 p3 kommt aus DET
        Q_det_vals
    )
    print("\n===== OPTIMIERUNG =====")
    print(f"MC  : r1={best_mc[0]:.1f}, r2={best_mc[1]:.1f}, r3={best_mc[2]:.1f}, val={val_mc:.5f}")
    print(f"DET : r1={best_det[0]:.1f}, r2={best_det[1]:.1f}, r3={best_det[2]:.1f}, val={val_det:.5f}")

        # Exaktes r3
    r3_det, _ = find_r3_det(sigma)
    
    # Kontinuierliche Optimierung
    r1_opt, r2_opt, val_opt = optimize_r1_r2_det(
        rs,
        P_det_vals,
        Q_det_vals,
        r3_det
    )
    
    print("\n===== KONTINUIERLICHE OPTIMIERUNG =====")
    print(f"r1 = {r1_opt:.4f}")
    print(f"r2 = {r2_opt:.4f}")
    print(f"r3 = {r3_det:.4f}")
    print(f"Checkout-Wkeit = {val_opt:.6f}")

 
    # -----------------------------
    # Heatmaps
    # -----------------------------
    plot_r1_r2_heatmap(rs, P, Q)                # MC
    r3_det, p3_det = find_r3_det(sigma)
    %matplotlib widget
    plot_r1_r2_heatmap_det(
        rs,
        P_det_vals,
        Q_det_vals,
        r3_det   # 🔥 jetzt korrekt
    )    
    # Optional direkter Vergleich
    #compare_heatmaps(rs, P, Q, P_det_vals, Q_det_vals)
