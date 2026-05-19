import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# -----------------------------
# Board-Parameter (mm)
# -----------------------------
R_DOUBLE_INNER = 162
R_DOUBLE_OUTER = 170

segments = np.array([
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
])

ANGLE_PER_SEG = 2 * np.pi / 20


# -----------------------------
# Deterministisches p(x,y)
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
        val, _ = quad(integrand, theta_min, theta_max,
                      args=(rho,), epsabs=1e-5, epsrel=1e-5)
        return rho * val

    val, _ = quad(integrand_rho,
                  R_DOUBLE_INNER, R_DOUBLE_OUTER,
                  epsabs=1e-5, epsrel=1e-5)

    return val / (2*np.pi*sigma**2)


# -----------------------------
# Zoom-Heatmap um D1
# -----------------------------
def plot_zoom_p_heatmap(sigma, resolution=200):

    theta_center = np.pi/2 - (1 + 0.5)*ANGLE_PER_SEG + np.pi/20
    r_center = (R_DOUBLE_INNER + R_DOUBLE_OUTER) / 2

    x_center = r_center * np.cos(theta_center)
    y_center = r_center * np.sin(theta_center)

    zoom = 30

    xs = np.linspace(x_center - zoom, x_center + zoom, resolution)
    ys = np.linspace(y_center - zoom, y_center + zoom, resolution)

    P = np.zeros((resolution, resolution))

    print("Berechne hochaufgelöste p-Heatmap...")

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            P[j, i] = p_det_xy(x, y, sigma)

    # Maximum
    idx = np.unravel_index(np.argmax(P), P.shape)
    x_max = xs[idx[1]]
    y_max = ys[idx[0]]
    p_max = P[idx]

    print(f"Maximum bei: ({x_max:.2f}, {y_max:.2f},{np.sqrt(x_max**2+y_max**2)}, {np.arctan(y_max/x_max)*np.pi/180}")
    print(f"Maximale W-Keit: {p_max:.6f}")

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        P,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin='lower'
    )

    plt.colorbar(im, ax=ax)

    # Maximum markieren
    ax.scatter(x_max, y_max, color='red', s=50, label='Maximum')
    ax.legend()

    # -----------------------------
    # Segmentlinien hinzufügen
    # -----------------------------
    EXT = 200
    
    for i in range(20):
        angle = np.pi/2 - i * ANGLE_PER_SEG + np.pi/20
    
        x_line = EXT * np.cos(angle)
        y_line = EXT * np.sin(angle)
    
        ax.plot([0, x_line], [0, y_line],
                color='white',
                linewidth=0.6,
                alpha=0.8)
    
    # 🔥 WICHTIG: Zoom wieder herstellen
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    # Optional: Double-Ring zur Orientierung
    circle_inner = plt.Circle((0, 0), R_DOUBLE_INNER,
                             fill=False, color='white', linewidth=0.8)
    circle_outer = plt.Circle((0, 0), R_DOUBLE_OUTER,
                             fill=False, color='white', linewidth=0.8)

    ax.add_artist(circle_inner)
    ax.add_artist(circle_outer)

    ax.set_title("Zoom auf D1: P(Check D1)")

    ax.set_aspect('equal')
    ax.axis('off')

    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    sigma = 40

    plot_zoom_p_heatmap(
        sigma=sigma,
        resolution=100
    )
