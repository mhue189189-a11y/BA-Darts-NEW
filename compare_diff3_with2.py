# ============================================================
# PLOT NUR DER DIFFERENZEN
#
# Absolute und relative Differenz zwischen
#
# Optimal/Greedy
# und
# Naiver Strategie
#
# für:
#
# p1(r1) * pD1(r2)
# ============================================================

def plot_strategy_differences():

    sigmas = np.linspace(3, 80, 60)

    abs_diffs = []
    rel_diffs = []

    # --------------------------------------------------------
    # Naive Strategie
    # --------------------------------------------------------

    r1_naive = (R_TRIPLE_OUTER + R_DOUBLE_INNER)/2
    r2_naive = 166

    for sigma in sigmas:

        # ----------------------------------------------------
        # Optimal / Greedy
        # ----------------------------------------------------

        r1_opt = find_r1_det(sigma)
        r2_opt = find_r3_det(sigma)

        val_opt = (
            p1_det(r1_opt, sigma)
            * p_det(r2_opt, sigma)
        )

        # ----------------------------------------------------
        # Naiv
        # ----------------------------------------------------

        val_naive = (
            p1_det(r1_naive, sigma)
            * p_det(r2_naive, sigma)
        )

        # ----------------------------------------------------
        # Differenzen
        # ----------------------------------------------------

        abs_diff = val_opt - val_naive

        rel_diff = (
            abs_diff / val_naive
            * 100
        )

        abs_diffs.append(abs_diff)
        rel_diffs.append(rel_diff)

    abs_diffs = np.array(abs_diffs)
    rel_diffs = np.array(rel_diffs)

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(9,6))

    plt.plot(
        sigmas,
        abs_diffs,
        linewidth=2,
        label="Absolute Differenz"
    )


    

    plt.xlabel(r"$\sigma$ (mm)")
    plt.ylabel("Differenz")

    plt.title(
        "Differenz zwischen optimaler und naiver Strategie"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    plot_strategy_differences()
