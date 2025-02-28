import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Callable


class ReacDiffSolver:
    """Solving the linear diffusion-reaction equation

        u_t = mu u_xx + a u

    using a modified Crank Nicholson scheme on the domain
    0 <= x <= 1, 0 <= t <= T, with boundary conditions
    u(0, x) = f(x), u(t, 0) = g1(t), u(t, 1) = g2(t).
    """

    def __init__(
        self,
        mu: float,
        a: float,
        f: Callable,
        g1: Callable,
        g2: Callable,
        alpha: float = 1,
    ):
        self.mu = mu
        self.a = a
        self.f = f
        self.g1 = g1
        self.g2 = g2
        self.alpha = alpha

    def solve(
        self, M: int, N: int, T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M: number of inner grid points in space.
        N: number of steps in time.
        T: end time

        returns:
            array of size (N+1, M+2)
            array with points in space (M+2)
            array with points in time (N+1)
        """
        k = T / N  # step size in time
        h = 1 / (M + 1)  # step size in space
        r = self.mu * k / h**2

        sol = np.zeros((N + 1, M + 2))
        x = np.linspace(0, 1, M + 2)
        t = np.linspace(0, T, N + 1)

        # set boundaries
        sol[0, :] = self.f(x)
        sol[1:, 0] = self.g1(t[1:])
        sol[1:, -1] = self.g2(t[1:])

        # construct the lhs of the system: A U* = b
        A = sp.sparse.diags(
            [-r / 2, 1 + r, -r / 2], offsets=[-1, 0, 1], shape=(M, M)
        ).tocsr()

        for n in range(N):
            Unm1 = sol[n, :-2]  # U_{n-1}
            Un = sol[n, 1:-1]   # U_{n}
            Unp1 = sol[n, 2:]   # U_{n+1}

            # solve equation for U* (implicit step)
            b = (1 + k * self.a) * Un + r / 2 * (Unm1 - 2 * Un + Unp1)
            b[0] += r / 2 * self.g1(t[n] + self.alpha * k)   # how to choose U_0^*?
            b[-1] += r / 2 * self.g2(t[n] + self.alpha * k)  # how to choose U_{M+1}^*?

            Ustar = sp.sparse.linalg.spsolve(A, b)

            # do a step forward in time (explicit step)
            sol[n + 1, 1:-1] = Ustar + k / 2 * (self.a * Ustar - self.a * Un)

        return sol, x, t


class ReacDiffTester:
    N_const = M_const = 10_000
    NUM_POINTS = 5

    def __init__(self, mu: float, a: float, b: float, phi: float, T: float = 1):
        self.mu = mu
        self.a = a
        self.b = b
        self.phi = phi
        self.T = T

        self.solver = ReacDiffSolver(
            mu=self.mu,
            a=self.a,
            f=lambda x: self.analytic(0, x),
            g1=lambda t: self.analytic(t, 0),
            g2=lambda t: self.analytic(t, 1),
            alpha=1,
        )

        self.analytic = lambda t, x: np.exp(
            (-self.mu * self.b**2 + self.a) * t
        ) * np.sin(self.b * x + self.phi)

    def generate_convergence_plots(self, filename: str):
        """Generates three convergence plots, where
        the global error is plotted against
        h (whilst keeping k constant),
        k (whilst keeping h constant) and
        h (whilst keeping k = h)
        """
        Ms = np.logspace(1, 3, self.NUM_POINTS, dtype=int) + 1
        Ns = np.logspace(1, 3, self.NUM_POINTS, dtype=int)

        errs = {
            "h": np.zeros(Ms.size, dtype=float),    # keep k constant and let h -> 0
            "k": np.zeros(Ms.size, dtype=float),    # keep h constant and let k -> 0
            "hk": np.zeros(Ms.size, dtype=float),   # let h = k -> 0
        }

        for i in range(self.NUM_POINTS):
            for c, _M, _N in (
                ("h", Ms[i], self.N_const),
                ("k", self.M_const, Ns[i]),
                ("hk", Ms[i], Ns[i]),
            ):
                sol, x, t = self.solver.solve(M=_M, N=_N, T=self.T)
                anal = self.analytic(*np.meshgrid(t, x)).T
                errs[c][i] = np.max(
                    1 / np.sqrt(x.size) * np.linalg.norm(sol - anal, ord=2, axis=1)
                )

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))

        fig.suptitle("Convergence plots")

        axs[0].loglog(1 / Ms, errs["h"], ".-m", label=f"k = {self.T/self.N_const}")
        axs[0].loglog(1 / Ms, (1 / Ms) ** 2, "--", label=r"$\mathcal{O}(h^2)$")
        axs[0].set(
            xlabel=r"$h$",
            ylabel=r"$||E^n||_{2,h}$",
            title=r"Constant $k$",
        )

        axs[1].loglog(self.T / Ns, errs["k"], ".-r", label=f"h = {1/self.M_const}")
        axs[1].loglog(self.T / Ns, (self.T / Ns) ** 2, "--", label=r"$\mathcal{O}(k^2)$")
        axs[1].set(
            xlabel=r"$k$",
            ylabel=r"$||E^n||_{2,h}$",
            title=r"Constant $h$",
        )

        axs[2].loglog(self.T / Ns, errs["hk"], ".-g", label="h=k")
        axs[2].loglog(self.T / Ns, (self.T / Ns) ** 2, "--", label=r"$\mathcal{O}{(h^2)}$")
        axs[2].loglog(self.T / Ns, 0.005 * self.T / Ns, "--", label=r"$\mathcal{O}{(h)}$")
        axs[2].set(
            xlabel=r"$h,k$",
            ylabel=r"$||E^n||_{2,h}$",
            title=r"$(h,k) \to (0,0)$ along $h=k$",
        )

        for i in (0, 1, 2):
            axs[i].legend()

        fig.tight_layout()
        fig.savefig(f"{filename}")


def plot3d(x, t, sol):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*np.meshgrid(x, t), sol)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.show()


def task1():
    mu = 1 / 5
    a = 1
    b = 3 / 2 * np.pi
    phi = 1 / 4 * np.pi
    T = 1

    tester = ReacDiffTester(mu=mu, a=a, b=b, phi=phi, T=T)
    tester.generate_convergence_plots("task1_error.pdf")


def main():
    task1()


if __name__ == "__main__":
    main()
