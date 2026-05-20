# ============================================================
# CHECKOUT 3
# Abbildungen:
# 1) p1(r), p0(r) entlang S1
# 2) Heatmap p1(x,y)
# 3) Heatmap p0(x,y)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# ============================================================
# BOARD
# ============================================================

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

ANGLE_PER_SEG = 2*np.pi/20


# ============================================================
# WINKEL
# ============================================================

def get_angle_for_segment(seg_value):

    i = np.where(segments == seg_value)[0][0]

    return np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20


# ============================================================
# SCORE
# ============================================================

def score_point(x, y):

    r = np.sqrt(x**2 + y**2)

    # außerhalb
    if r > R_DOUBLE_OUTER:
        return 0

    theta = np.arctan2(y, x)
    theta = (np.pi/2 - theta + np.pi/20) % (2*np.pi)

    seg_index = int(np.floor(theta / ANGLE_PER_SEG))
    seg_index = np.clip(seg_index, 0, 19)

    base = segments[seg_index]

    # Bull
    if r <= R_BULL_INNER:
        return 50

    if r <= R_BULL_OUTER:
        return 25

    # Triple
    if R_TRIPLE_INNER <= r <= R_TRIPLE_OUTER:
        return 3 * base

    # Double
    if R_DOUBLE_INNER <= r <= R_DOUBLE_OUTER:
        return 2 * base

    return base


# ============================================================
# MONTE CARLO
# ============================================================

def compute_p1_p0(center, sigma, samples=20000):

    x = np.random.normal(center[0], sigma, samples)
    y = np.random.normal(center[1], sigma, samples)

    scores = np.array([
        score_point(xi, yi)
        for xi, yi in zip(x, y)
    ])

    p1 = np.mean(scores == 1)
    p0 = np.mean(scores == 0)

    return p1, p0


# ============================================================
# PLOT p1(r), p0(r)
# ============================================================

def plot_radial_probabilities(sigma=10):

    angle = get_angle_for_segment(1)

    rs = np.linspace(0, 250, 80)

    P1 = []
    P0 = []

    print("Berechne p1(r), p0(r)...")

    for r in rs:

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        p1, p0 = compute_p1_p0((x, y), sigma)

        P1.append(p1)
        P0.append(p0)

    P1 = np.array(P1)
    P0 = np.array(P0)

    plt.figure(figsize=(8,5))

    plt.plot(rs, P1, label=r"$p_1(r)$")
    plt.plot(rs, P0, label=r"$p_0(r)$")

    plt.xlabel("Radius r (mm)")
    plt.ylabel("Wahrscheinlichkeit")

    plt.title(
        rf"$p_1(r)$ und $p_0(r)$ entlang der Symmetrieachse von $S1$ ($\sigma={sigma}$)"
    )

    plt.grid()
    plt.legend()

    plt.tight_layout()

    plt.savefig("p1_p0_vs_radius.png", dpi=300)

    plt.show()


# ============================================================
# SEGMENTLINIEN + ZAHLENRING FÜR DIE HEATMAPS
# ============================================================

def draw_board_overlay():

    ax = plt.gca()

    # --------------------------------------------------------
    # Ringe
    # --------------------------------------------------------

    rings = [
        R_BULL_INNER,
        R_BULL_OUTER,
        R_TRIPLE_INNER,
        R_TRIPLE_OUTER,
        R_DOUBLE_INNER,
        R_DOUBLE_OUTER
    ]

    for r in rings:

        circle = plt.Circle(
            (0, 0),
            r,
            fill=False,
            color='black',
            linewidth=0.6
        )

        ax.add_patch(circle)

    # --------------------------------------------------------
    # Segmentlinien
    # --------------------------------------------------------

    for k in range(20):

        theta = np.pi/2+np.pi/20 - k * ANGLE_PER_SEG

        x1 = R_BULL_OUTER * np.cos(theta)
        y1 = R_BULL_OUTER * np.sin(theta)

        x2 = R_DOUBLE_OUTER * np.cos(theta)
        y2 = R_DOUBLE_OUTER * np.sin(theta)

        plt.plot(
            [x1, x2],
            [y1, y2],
            color='black',
            linewidth=0.4
        )

    # --------------------------------------------------------
    # Zahlenring
    # --------------------------------------------------------

    number_radius = R_DOUBLE_OUTER + 14

    for k in range(20):

        theta = np.pi/2+np.pi/20 - (k + 0.5) * ANGLE_PER_SEG

        x = number_radius * np.cos(theta)
        y = number_radius * np.sin(theta)

        plt.text(
            x,
            y,
            str(segments[k]),
            fontsize=8,
            ha='center',
            va='center',
            fontweight='bold'
        )


# ============================================================
# AKTUALISIERTE HEATMAP-FUNKTION
# ============================================================

def heatmap_probability(
    sigma=10,
    mode="p1",
    resolution=80
):

    xs = np.linspace(-200, 200, resolution)
    ys = np.linspace(-200, 200, resolution)

    Z = np.zeros((resolution, resolution))

    print(f"Berechne Heatmap {mode} ...")

    samples = 300

    for i, x0 in enumerate(xs):

        for j, y0 in enumerate(ys):

            x = np.random.normal(x0, sigma, samples)
            y = np.random.normal(y0, sigma, samples)

            scores = np.array([
                score_point(xi, yi)
                for xi, yi in zip(x, y)
            ])

            if mode == "p1":
                Z[j, i] = np.mean(scores == 1)

            elif mode == "p0":
                Z[j, i] = np.mean(scores == 0)

    plt.figure(figsize=(8,8))

    plt.imshow(
        Z,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin='lower'
    )

    plt.colorbar()

    # --------------------------------------------------------
    # DARTBOARD OVERLAY
    # --------------------------------------------------------

    draw_board_overlay()

    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    if mode == "p1":

        plt.title(
            rf"Heatmap von $p_1(x,y)$ ($\sigma={sigma}$)"
        )

        plt.savefig("heatmap_p1.png", dpi=300)

    else:

        plt.title(
            rf"Heatmap von $p_0(x,y)$ ($\sigma={sigma}$)"
        )

        plt.savefig("heatmap_p0.png", dpi=300)

    plt.axis('equal')

    plt.tight_layout()

    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    sigma = 10

    # --------------------------------------------------------
    # 1) p1(r), p0(r)
    # --------------------------------------------------------

    plot_radial_probabilities(sigma=sigma)

    # --------------------------------------------------------
    # 2) Heatmap p1
    # --------------------------------------------------------

    heatmap_probability(
        sigma=sigma,
        mode="p1"
    )

    # --------------------------------------------------------
    # 3) Heatmap p0
    # --------------------------------------------------------

    heatmap_probability(
        sigma=sigma,
        mode="p0"
    )
