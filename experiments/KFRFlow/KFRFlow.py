import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# define the log density ratio (ldr)
ldr = lambda x: -8 * (np.linalg.norm(x) - 2)**2

# define the kernel and the kernel gradient for reference
kernel = lambda x, y: (1 + np.sqrt(3) / 45 * np.linalg.norm(x - y)) * np.exp(- np.sqrt(3) / 45 * np.linalg.norm(x - y))
nabla_kernel = lambda x, y: -3 / 45**2 * np.exp(-np.sqrt(3) / 45 * np.linalg.norm(x - y)) * (x - y) 


class donut:
    def __init__(self, J=300, d=2, seed=0):
        self.J = J
        self.d = d
        rng = np.random.default_rng(seed)
        # sample 300 2D Gaussian r.v.s
        self.X = rng.standard_normal((J, d))
        self.data = [self.X.copy()]
        self.times = [0.0]

    def get_velocity_field(self):
        # construct b: b is K_{pi_t} (ldr - ldr_bar) (RHS of equation (6))
        # we'll have to do this for all collocation points/kernel sections/particles/rows simultaneously
        # we need \int_{\R^d} K(\cdot, x) (ldr(x) - ldr_bar) d\pi_t(x)
        # using the samples as an empirical measure, the integral is replaced by the empirical average at argument x

        # generate diff tensor
        diff_tensor = self.X[:, None, :] - self.X[None, :, :]
        diff_norms = np.linalg.norm(diff_tensor, axis=2)

        # get Kernel matrix
        K = (1 + np.sqrt(3) / 45 * diff_norms) * np.exp(- np.sqrt(3) / 45 * diff_norms)

        # get ldr - ldr_bar
        q = -8 * (np.linalg.norm(self.X, axis=1) - 2)**2
        q -= q.mean()

        # integrate ldr weighted by the Kernel matrix
        b = K @ q

        # next we construct M
        A = (-3 / 45**2 * np.exp(-np.sqrt(3) / 45 * diff_norms))[:, :, None] * (diff_tensor)
        M = np.einsum('ijk,ilk->jl', A, A)

        # finally we solve for alpha
        alpha = np.linalg.solve(M + 1e-8 * np.eye(self.J), b)

        # integrate to get V
        V = np.einsum('ijk,j->ik', A, alpha)
        return V

    def forward_euler(self, delta_t=0.01, steps=100):
        for step in range(1, steps + 1):
            self.X = self.X + delta_t * self.get_velocity_field()
            if step % 25 == 0 or step == steps:
                self.data.append(self.X.copy())
                self.times.append(step * delta_t)

    def plot(self, t=1.0, xmin=-3.0, xmax=3.0, points=300):
        
        if self.d != 2:
            raise ValueError("The donut plot requires d=2.")

        x1 = np.linspace(xmin, xmax, points)
        x2 = np.linspace(xmin, xmax, points)
        X1, X2 = np.meshgrid(x1, x2)
        radius = np.sqrt(X1**2 + X2**2)

        log_density = -0.5 * radius**2 - 8 * t * (radius - 2)**2
        density = np.exp(log_density)

        fig, ax = plt.subplots()
        ax.contourf(X1, X2, density, levels=8)
        ax.scatter(self.X[:, 0], self.X[:, 1], color="red", alpha=0.5, s=15)
        ax.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            title=f"KFRFlow, Euler, t={t:g}",
            xlim=(xmin, xmax),
            ylim=(xmin, xmax),
        )
        ax.set_aspect("equal")
        return fig, ax

    def plot_history(self, xmin=-3.0, xmax=3.0, points=300):
        if self.d != 2:
            raise ValueError("The donut plot requires d=2.")

        x1 = np.linspace(xmin, xmax, points)
        x2 = np.linspace(xmin, xmax, points)
        X1, X2 = np.meshgrid(x1, x2)
        radius = np.sqrt(X1**2 + X2**2)

        fig, axes = plt.subplots(
            1, len(self.data), figsize=(4 * len(self.data), 4), sharex=True, sharey=True
        )

        for ax, samples, t in zip(axes, self.data, self.times):
            log_density = -0.5 * radius**2 - 8 * t * (radius - 2)**2
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


if __name__ == "__main__":
    model = donut()
    model.forward_euler()
    fig, _ = model.plot()
    output_path = Path(__file__).with_name("donut.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")

    history_fig, _ = model.plot_history()
    history_path = Path(__file__).with_name("donut_history.png")
    history_fig.savefig(history_path, dpi=200, bbox_inches="tight")
    plt.close(history_fig)
    print(f"Saved plot history to {history_path}")
    
