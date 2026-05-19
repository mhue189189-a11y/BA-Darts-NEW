import numpy as np
import matplotlib.pyplot as plt

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


# -----------------------------
# p(x,y) und q(x,y)
# -----------------------------
def compute_p_q(center, sigma, samples=30000):

    x = np.random.normal(center[0], sigma, samples)
    y = np.random.normal(center[1], sigma, samples)

    r = np.sqrt(x**2 + y**2)

    # Winkel
    theta = np.arctan2(y, x)
    theta = (np.pi/2 - theta + np.pi/20) % (2*np.pi)

    seg_index = np.floor(theta / ANGLE_PER_SEG).astype(int)
    seg_index = np.clip(seg_index, 0, 19)
    base = segments[seg_index]

    # Treffer auf D1
    hit_d1 = (
        (r >= R_DOUBLE_INNER) & (r <= R_DOUBLE_OUTER) &
        (base == 1)
    )

    # Scores berechnen
    scores = np.zeros_like(r)

    # Bull
    scores[r <= R_BULL_INNER] = 50

    mask_outer = (r > R_BULL_INNER) & (r <= R_BULL_OUTER)
    scores[mask_outer] = 25

    # Triple
    mask_triple = (r >= R_TRIPLE_INNER) & (r <= R_TRIPLE_OUTER)
    scores[mask_triple] = 3 * base[mask_triple]

    # Double
    mask_double = (r >= R_DOUBLE_INNER) & (r <= R_DOUBLE_OUTER)
    scores[mask_double] = 2 * base[mask_double]

    # Single
    mask_single = (r > R_BULL_OUTER) & (r < R_DOUBLE_INNER)

    scores[
        mask_single & ~mask_triple & ~mask_double
    ] = base[
        mask_single & ~mask_triple & ~mask_double
    ]

    # Bust-Regel bei Rest = 2
    bust = (scores > 0) | ((scores == 2) & (~hit_d1))

    p = np.mean(hit_d1)
    q = np.mean(bust)

    return p, q


# -----------------------------
# Heatmaps
# -----------------------------
def plot_pq_heatmaps(sigma, resolution=120, samples=20000):

    # ---------------------------------
    # FIXES QUADRAT: 400 x 400 mm
    # ---------------------------------
    HALF_SIZE = 200

    xs = np.linspace(-HALF_SIZE, HALF_SIZE, resolution)
    ys = np.linspace(-HALF_SIZE, HALF_SIZE, resolution)

    P = np.zeros((resolution, resolution))
    Q = np.zeros((resolution, resolution))

    print("Berechne Heatmaps...")

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            p, q = compute_p_q((x, y), sigma, samples=samples)

            P[j, i] = p
            Q[j, i] = q

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    extent = [
        -HALF_SIZE,
        HALF_SIZE,
        -HALF_SIZE,
        HALF_SIZE
    ]

    im1 = ax[0].imshow(
        P,
        extent=extent,
        origin='lower'
    )

    ax[0].set_title("P(Check D1)")
    plt.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(
        Q - P,
        extent=extent,
        origin='lower'
    )

    ax[1].set_title("Q(Bust)")
    plt.colorbar(im2, ax=ax[1])

    # -----------------------------
    # Dartboard Overlay
    # -----------------------------
    for a in ax:

        # Ringe
        for r in [
            R_BULL_INNER,
            R_BULL_OUTER,
            R_TRIPLE_INNER,
            R_TRIPLE_OUTER,
            R_DOUBLE_INNER,
            R_DOUBLE_OUTER
        ]:
            a.add_artist(
                plt.Circle(
                    (0, 0),
                    r,
                    fill=False,
                    color='white',
                    linewidth=0.6
                )
            )

        # Segmentlinien
        for i in range(20):

            angle = np.pi/2 - i * ANGLE_PER_SEG + np.pi/20

            x = HALF_SIZE * np.cos(angle)
            y = HALF_SIZE * np.sin(angle)

            a.plot(
                [0, x],
                [0, y],
                color='white',
                linewidth=0.35,
                alpha=0.7
            )

        # Zahlenring
        for i, num in enumerate(segments):

            angle = np.pi/2 - (i + 0.5) * ANGLE_PER_SEG + np.pi/20

            r_text = R_DOUBLE_OUTER + 20

            x = r_text * np.cos(angle)
            y = r_text * np.sin(angle)

            a.text(
                x,
                y,
                str(num),
                color='white',
                fontsize=8,
                ha='center',
                va='center',
                weight='bold'
            )

        # ---------------------------------
        # SCHARFER QUADRATISCHER RAND
        # ---------------------------------
        a.set_xlim(-HALF_SIZE, HALF_SIZE)
        a.set_ylim(-HALF_SIZE, HALF_SIZE)

        a.set_aspect('equal')
        a.axis('off')

    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    sigma = 5

    plot_pq_heatmaps(
        sigma=sigma,
        resolution=120,
        samples=20000
    )
