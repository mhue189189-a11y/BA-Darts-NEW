import numpy as np
import matplotlib.pyplot as plt

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
# MC-Modell für p(x,y)
# -----------------------------
def p_mc_xy(center, sigma, samples=20000):

    x = np.random.normal(center[0], sigma, samples)
    y = np.random.normal(center[1], sigma, samples)

    r = np.sqrt(x**2 + y**2)

    theta = np.arctan2(y, x)
    theta = (np.pi/2 - theta + np.pi/20) % (2*np.pi)

    seg_index = np.floor(theta / ANGLE_PER_SEG).astype(int)
    seg_index = np.clip(seg_index, 0, 19)
    base = segments[seg_index]

    hit_d1 = (
        (r >= R_DOUBLE_INNER) &
        (r <= R_DOUBLE_OUTER) &
        (base == 1)
    )

    return np.mean(hit_d1)


# -----------------------------
# Zoom-Heatmap (MC)
# -----------------------------
def plot_zoom_mc_heatmap(sigma, resolution=120, samples=15000):

    # Zentrum von D1
    theta_center = np.pi/2 - (1 + 0.5)*ANGLE_PER_SEG + np.pi/20
    r_center = (R_DOUBLE_INNER + R_DOUBLE_OUTER) / 2

    x_center = r_center * np.cos(theta_center)
    y_center = r_center * np.sin(theta_center)

    zoom = 30

    xs = np.linspace(x_center - zoom, x_center + zoom, resolution)
    ys = np.linspace(y_center - zoom, y_center + zoom, resolution)

    P = np.zeros((resolution, resolution))

    print("Berechne MC-Heatmap (Zoom)...")

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            P[j, i] = p_mc_xy((x, y), sigma, samples=samples)

    # Maximum
    idx = np.unravel_index(np.argmax(P), P.shape)
    x_max = xs[idx[1]]
    y_max = ys[idx[0]]
    p_max = P[idx]

    print(f"Maximum bei: ({x_max:.2f}, {y_max:.2f},{np.sqrt(x_max**2+y_max**2)}, {np.arctan(y_max/x_max)*np.pi/180})")
    print(f"Maximale W-Keit (MC): {p_max:.6f}")

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
    # Segmentlinien
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
    # Double-Ring
    ax.add_artist(plt.Circle((0, 0), R_DOUBLE_INNER,
                             fill=False, color='white', linewidth=0.8))
    ax.add_artist(plt.Circle((0, 0), R_DOUBLE_OUTER,
                             fill=False, color='white', linewidth=0.8))

    ax.set_title("MC Zoom auf D1: P(Check D1)")

    ax.set_aspect('equal')
    ax.axis('off')

    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    sigma = 40

    plot_zoom_mc_heatmap(
        sigma=sigma,
        resolution=80,   # höher = langsamer!
        samples=15000     # Samples pro Punkt
    )
