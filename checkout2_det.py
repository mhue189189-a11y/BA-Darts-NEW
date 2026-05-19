import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

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
# Deterministische Modelle
# -----------------------------
def p_det_xy(x0, y0, sigma):

    r0 = np.sqrt(x0**2 + y0**2)
    theta0 = np.arctan2(y0, x0)

    theta_center = np.pi/2 - (1 + 0.5)*ANGLE_PER_SEG + np.pi/20

    theta_min = theta_center - ANGLE_PER_SEG/2
    theta_max = theta_center + ANGLE_PER_SEG/2

    def integrand(theta, rho):

        return np.exp(
            -(rho**2 + r0**2 - 2*rho*r0*np.cos(theta - theta0))
            / (2*sigma**2)
        )

    def integrand_rho(rho):

        val, _ = quad(
            integrand,
            theta_min,
            theta_max,
            args=(rho,),
            epsabs=1e-5,
            epsrel=1e-5
        )

        return rho * val

    val, _ = quad(
        integrand_rho,
        R_DOUBLE_INNER,
        R_DOUBLE_OUTER,
        epsabs=1e-5,
        epsrel=1e-5
    )

    return val / (2*np.pi*sigma**2)


def Q_det_xy(x0, y0, sigma):

    r0 = np.sqrt(x0**2 + y0**2)

    def integrand_theta(theta, rho):

        return np.exp(
            -(rho**2 + r0**2 - 2*rho*r0*np.cos(theta))
            / (2*sigma**2)
        ) / (2*np.pi*sigma**2)

    def integrand_rho(rho):

        val, _ = quad(
            integrand_theta,
            0,
            2*np.pi,
            args=(rho,),
            epsabs=1e-5,
            epsrel=1e-5
        )

        return rho * val

    val, _ = quad(
        integrand_rho,
        0,
        R_DOUBLE_OUTER,
        epsabs=1e-5,
        epsrel=1e-5
    )

    return val


def q_det_xy(x0, y0, sigma):

    return Q_det_xy(x0, y0, sigma) - p_det_xy(x0, y0, sigma)


# -----------------------------
# Heatmap (Deterministisch)
# -----------------------------
def plot_det_heatmaps(sigma, resolution=60):

    # ---------------------------------
    # FESTES 400x400 mm QUADRAT
    # ---------------------------------
    HALF_SIZE = 200

    xs = np.linspace(-HALF_SIZE, HALF_SIZE, resolution)
    ys = np.linspace(-HALF_SIZE, HALF_SIZE, resolution)

    P = np.zeros((resolution, resolution))
    Q = np.zeros((resolution, resolution))

    print("Berechne deterministische Heatmaps...")

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            p = p_det_xy(x, y, sigma)
            q = q_det_xy(x, y, sigma)

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
        Q,
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

    plot_det_heatmaps(
        sigma=sigma,
        resolution=100   # sonst sehr langsam
    )
