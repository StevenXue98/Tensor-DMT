import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensor_dmt.kfrflow.flow import FlowResult, kfrflow_euler
from tensor_dmt.kfrflow.hermite import hfrflow_euler
from tensor_dmt.kfrflow.kernels import matern32_kernel


def donut_log_ratio(X: np.ndarray) -> np.ndarray:
    return -8.0 * (np.linalg.norm(X, axis=-1) - 2.0) ** 2


def _donut_grid(xmin, xmax, points):
    coordinates = np.linspace(xmin, xmax, points)
    x1, x2 = np.meshgrid(coordinates, coordinates)
    return x1, x2, np.hypot(x1, x2)


def _plot_particles(ax, samples, grid, t):
    x1, x2, radius = grid
    density = np.exp(-0.5 * radius**2 - 8 * t * (radius - 2)**2)
    ax.contourf(x1, x2, density, levels=8)
    finite = np.isfinite(samples).all(axis=1)
    ax.scatter(samples[finite, 0], samples[finite, 1], color="red", alpha=0.5, s=10)
    if not finite.all():
        ax.text(0.5, 0.5, f"{finite.sum()}/{len(samples)} finite particles",
                transform=ax.transAxes, ha="center", color="white")
    ax.set(xlim=(x1.min(), x1.max()), ylim=(x2.min(), x2.max()), aspect="equal")


def plot_donut(results: list[FlowResult], xmin=-3.0, xmax=3.0, points=300):
    grid = _donut_grid(xmin, xmax, points)
    fig, axes = plt.subplots(1, len(results), figsize=(3.5 * len(results), 4),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax, result in zip(axes[0], results):
        t = result.times[-1]
        _plot_particles(ax, result.final_particles, grid, t)
        ax.set(title=f"{result.method_name}\n{result.elapsed_seconds:.3f} s, t={t:g}",
               xlabel=r"$x_1$")
    axes[0, 0].set_ylabel(r"$x_2$")
    return fig, axes[0]


def plot_donut_history(results: list[FlowResult], xmin=-3.0, xmax=3.0, points=300):
    times = results[0].times
    grid = _donut_grid(xmin, xmax, points)
    fig, axes = plt.subplots(len(results), len(times), figsize=(3 * len(times), 3 * len(results)),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for row, result in zip(axes, results):
        if result.times != times:
            raise ValueError("All histories must use the same snapshot times.")
        for ax, samples, t in zip(row, result.history, times, strict=True):
            _plot_particles(ax, samples, grid, t)
        row[0].set_ylabel(f"{result.method_name}\n{result.elapsed_seconds:.3f} s\n" + r"$x_2$")
    for col, t in enumerate(times):
        axes[0, col].set_title(f"t={t:g}")
        axes[-1, col].set_xlabel(r"$x_1$")
    return fig, axes


def main():
    rng = np.random.default_rng(0)
    X_initial = rng.standard_normal((300, 2))

    kernel_result = kfrflow_euler(
        X_initial,
        donut_log_ratio,
        matern32_kernel,
        nugget=1e-8,
        delta_t=0.01,
        steps=100,
        save_history=True,
        save_every=25,
    )

    results = [kernel_result]

    # X_initial = rng.standard_normal((3000, 2))
    for degree in (3, 4, 5, 6, 7, 8, 9):
        results.append(hfrflow_euler(
            X_initial,
            donut_log_ratio,
            max_degree=degree,
            reg_order=None,
            nugget=1e-8,
            delta_t=0.01,
            steps=100,
            total_degree=False,
            save_history=True,
            save_every=25,
        ))
    for result in results:
        print(f"{result.method_name}: {result.elapsed_seconds:.3f} s")

    fig, _ = plot_donut(results)
    output_path = Path(__file__).with_name("donut.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")

    history_fig, _ = plot_donut_history(results)
    history_path = Path(__file__).with_name("donut_history.png")
    history_fig.savefig(history_path, dpi=200, bbox_inches="tight")
    plt.close(history_fig)
    print(f"Saved plot history to {history_path}")


if __name__ == "__main__":
    main()
