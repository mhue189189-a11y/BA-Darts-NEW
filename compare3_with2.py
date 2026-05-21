# ============================================================
# ZUSATZPLOT:
#
# Vergleich von
#   p1(r)
#   p_D1(r)
#
# für festes sigma
#
# Zusätzlich:
#   exakte numerische Differenz
#   zwischen
#
#   optimal/greedy
#   und
#   naiver Strategie
# ============================================================

def compare_probabilities_fixed_sigma(sigma=10):

    rs = np.linspace(0, 200, 500)

    # --------------------------------------------------------
    # Wahrscheinlichkeiten
    # --------------------------------------------------------

    p1_vals = np.array([
        p1_det(r, sigma)
        for r in rs
    ])

    pD1_vals = np.array([
        p_det(r, sigma)
        for r in rs
    ])

    # --------------------------------------------------------
    # Optimale Strategie
    # --------------------------------------------------------

    r1_opt = find_r1_det(sigma)
    r2_opt = find_r3_det(sigma)

    p1_opt = p1_det(r1_opt, sigma)
    pD1_opt = p_det(r2_opt, sigma)

    val_opt = p1_opt * pD1_opt

    # --------------------------------------------------------
    # Naive Strategie
    # --------------------------------------------------------

    r1_naive = (R_TRIPLE_OUTER + R_DOUBLE_INNER)/2
    r2_naive = 166

    p1_naive = p1_det(r1_naive, sigma)
    pD1_naive = p_det(r2_naive, sigma)

    val_naive = p1_naive * pD1_naive

    # --------------------------------------------------------
    # Differenzen
    # --------------------------------------------------------

    abs_diff = val_opt - val_naive

    rel_diff = (
        (val_opt - val_naive)
        / val_naive
        * 100
    )

    # --------------------------------------------------------
    # AUSGABE
    # --------------------------------------------------------

    print("\n================================================")
    print(f"σ = {sigma}")
    print("================================================")

    print("\nOPTIMAL / GREEDY")

    print(f"r1_opt = {r1_opt:.4f} mm")
    print(f"r2_opt = {r2_opt:.4f} mm")

    print(f"p1_opt  = {p1_opt:.8f}")
    print(f"pD1_opt = {pD1_opt:.8f}")

    print(f"p1*pD1  = {val_opt:.8f}")

    print("\nNAIV")

    print(f"r1_naive = {r1_naive:.4f} mm")
    print(f"r2_naive = {r2_naive:.4f} mm")

    print(f"p1_naive  = {p1_naive:.8f}")
    print(f"pD1_naive = {pD1_naive:.8f}")

    print(f"p1*pD1    = {val_naive:.8f}")

    print("\nDIFFERENZEN")

    print(f"Absolute Differenz = {abs_diff:.8f}")
    print(f"Relative Differenz = {rel_diff:.4f} %")

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(10,6))

    # --------------------------------------------------------
    # p1
    # --------------------------------------------------------

    plt.plot(
        rs,
        p1_vals,
        linewidth=2,
        label=r"$p_1(r)$"
    )

    # --------------------------------------------------------
    # p_D1
    # --------------------------------------------------------

    plt.plot(
        rs,
        pD1_vals,
        linewidth=2,
        label=r"$p_{D1}(r)$"
    )

    # --------------------------------------------------------
    # Optimale Radien
    # --------------------------------------------------------

    plt.axvline(
        r1_opt,
        linestyle='--',
        linewidth=2,
        label=rf"Optimal: $r_1^*={r1_opt:.2f}$ mm"
    )

    plt.axvline(
        r2_opt,
        linestyle=':',
        linewidth=2,
        label=rf"Optimal: $r_2^*={r2_opt:.2f}$ mm"
    )

    # --------------------------------------------------------
    # Naive Radien
    # --------------------------------------------------------

    plt.axvline(
        r1_naive,
        linestyle='-.',
        linewidth=2,
        label=rf"Naiv: $r_1={r1_naive:.2f}$ mm"
    )

    plt.axvline(
        r2_naive,
        linestyle=(0, (1,1)),
        linewidth=2,
        label=rf"Naiv: $r_2={r2_naive:.2f}$ mm"
    )

    plt.xlabel("Radius r (mm)")
    plt.ylabel("Wahrscheinlichkeit")

    plt.title(
        rf"$p_1(r)$ und $p_{{D1}}(r)$ für $\sigma={sigma}$"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN ERGÄNZUNG
# ============================================================

if __name__ == "__main__":

    sigma = 10

    # --------------------------------------------------------
    # p1 Plot
    # --------------------------------------------------------

    plot_p1_det(sigma=sigma)

    # --------------------------------------------------------
    # Strategievergleich
    # --------------------------------------------------------

    compare_checkout_3_two_darts()

    # --------------------------------------------------------
    # DETAILVERGLEICH
    # --------------------------------------------------------

    compare_probabilities_fixed_sigma(sigma=sigma)
