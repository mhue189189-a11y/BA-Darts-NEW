# ============================================================
# PLOT: r3*(sigma)
# kontinuierlich deterministisch optimiert
# ============================================================

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
