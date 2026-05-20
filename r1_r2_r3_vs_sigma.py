# ============================================================
# PLOT: r1*(sigma), r2*(sigma), r3*(sigma)
# kontinuierliche deterministische Optimierung
# ============================================================
import numpy as np
sigmas = np.linspace(2, 80, 30)

r1_list = []
r2_list = []
r3_list = []

val_list = []

print("Berechne kontinuierliche Optimierung...")

for sigma in sigmas:

    print(f"\n===== sigma = {sigma:.2f} =====")

    # ------------------------------------------------
    # feines Referenzgitter
    # ------------------------------------------------
    rs = np.linspace(0, 250, 400)

    P_det_vals = np.array([
        p_det(r, sigma) for r in rs
    ])

    Q_det_vals = np.array([
        q_det(r, sigma) for r in rs
    ])

    # ------------------------------------------------
    # optimales r3
    # ------------------------------------------------
    r3_opt, p3_opt = find_r3_det(sigma)

    # ------------------------------------------------
    # kontinuierliche Optimierung von r1,r2
    # ------------------------------------------------
    r1_opt, r2_opt, val_opt = optimize_r1_r2_det(
        rs,
        P_det_vals,
        Q_det_vals,
        r3_opt
    )

    r1_list.append(r1_opt)
    r2_list.append(r2_opt)
    r3_list.append(r3_opt)

    val_list.append(val_opt)

    print(f"r1 = {r1_opt:.4f}")
    print(f"r2 = {r2_opt:.4f}")
    print(f"r3 = {r3_opt:.4f}")
    print(f"Checkout-Wkeit = {val_opt:.6f}")

# ------------------------------------------------
# Arrays
# ------------------------------------------------
r1_list = np.array(r1_list)
r2_list = np.array(r2_list)
r3_list = np.array(r3_list)

# ------------------------------------------------
# Plot
# ------------------------------------------------
plt.figure(figsize=(9,5))

plt.plot(sigmas, r1_list, label=r"$r_1^\ast$", linewidth=2)
plt.plot(sigmas, r2_list, label=r"$r_2^\ast$", linewidth=2)
plt.plot(sigmas, r3_list, label=r"$r_3^\ast$= r_greedy", linewidth=2)

plt.hlines(166,0,80, label="r_naive", linewidth=1, linestyles="-")

plt.xlabel(r"$\sigma$ (mm)")
plt.ylabel("Optimaler Radius (mm)")

plt.title(
    r"Kontinuierlich optimierte Radien "
    r"$r_1^\ast,r_2^\ast,r_3^\ast$ "
    r"in Abhängigkeit von $\sigma$"
)

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
