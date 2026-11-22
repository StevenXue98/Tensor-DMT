import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensor_dmt.kfrflow.flow import forward_euler
from tensor_dmt.kfrflow.kernels import matern32_kernel


def donut_log_ratio(X: np.ndarray) -> np.ndarray:
    return -8.0 * (np.linalg.norm(X, axis=-1) - 2.0) ** 2


def plot_donut(
    X: np.ndarray,
    t: float = 1.0,
    xmin: float = -3.0,
    xmax: float = 3.0,
    points: int = 300,
):
    if X.shape[1] != 2:
        raise ValueError("The donut plot requires d=2.")

    x1 = np.linspace(xmin, xmax, points)
    x2 = np.linspace(xmin, xmax, points)
    X1, X2 = np.meshgrid(x1, x2)
    radius = np.sqrt(X1**2 + X2**2)

    log_density = -0.5 * radius**2 - 8 * t * (radius - 2) ** 2
    density = np.exp(log_density)

    fig, ax = plt.subplots()
    ax.contourf(X1, X2, density, levels=8)
    ax.scatter(X[:, 0], X[:, 1], color="red", alpha=0.5, s=15)
    ax.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        title=f"KFRFlow, Euler, t={t:g}",
        xlim=(xmin, xmax),
        ylim=(xmin, xmax),
    )
    ax.set_aspect("equal")
    return fig, ax


def plot_donut_history(
    history: list[np.ndarray],
    times: list[float],
    xmin: float = -3.0,
    xmax: float = 3.0,
    points: int = 300,
):
    if history[0].shape[1] != 2:
        raise ValueError("The donut plot requires d=2.")

    x1 = np.linspace(xmin, xmax, points)
    x2 = np.linspace(xmin, xmax, points)
    X1, X2 = np.meshgrid(x1, x2)
    radius = np.sqrt(X1**2 + X2**2)

    fig, axes = plt.subplots(
        1,
        len(history),
        figsize=(4 * len(history), 4),
        sharex=True,
        sharey=True,
    )

    for ax, samples, t in zip(axes, history, times):
        log_density = -0.5 * radius**2 - 8 * t * (radius - 2) ** 2
        density = np.exp(log_density)
        ax.contourf(X1, X2, density, levels=8)
        ax.scatter(samples[:, 0], samples[:, 1], color="red", alpha=0.5, s=10)
        ax.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            title=f"t={t:g}",
            xlim=(xmin, xmax),
            ylim=(xmin, xmax),
        )
        ax.set_aspect("equal")

    fig.suptitle("KFRFlow, Euler")
    fig.tight_layout()
    return fig, axes


def main():
    rng = np.random.default_rng(0)
    X_initial = rng.standard_normal((300, 2))

    result = forward_euler(
        X_initial,
        donut_log_ratio,
        matern32_kernel,
        nugget=1e-8,
        delta_t=0.01,
        steps=100,
        save_history=True,
        save_every=25,
    )

    fig, _ = plot_donut(result.final_particles)
    output_path = Path(__file__).with_name("donut.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")

    history_fig, _ = plot_donut_history(result.history, result.times)
    history_path = Path(__file__).with_name("donut_history.png")
    history_fig.savefig(history_path, dpi=200, bbox_inches="tight")
    plt.close(history_fig)
    print(f"Saved plot history to {history_path}")


if __name__ == "__main__":
    main()
