import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Callable, Union
from pathlib import Path

PLOTS_PATH = Path("./Plots/")

#############################################
# ---------------- Task 1 -------------------
#############################################


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
            Un = sol[n, 1:-1]  # U_{n}
            Unp1 = sol[n, 2:]  # U_{n+1}

            # solve equation for U* (implicit step)
            b = (1 + k * self.a) * Un + r / 2 * (Unm1 - 2 * Un + Unp1)
            b[0] += r / 2 * self.g1(t[n] + self.alpha * k)  # how to choose U_0^*?
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
            "h": np.zeros(Ms.size, dtype=float),  # keep k constant and let h -> 0
            "k": np.zeros(Ms.size, dtype=float),  # keep h constant and let k -> 0
            "hk": np.zeros(Ms.size, dtype=float),  # let h = k -> 0
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
        axs[1].loglog(
            self.T / Ns, (self.T / Ns) ** 2, "--", label=r"$\mathcal{O}(k^2)$"
        )
        axs[1].set(
            xlabel=r"$k$",
            ylabel=r"$||E^n||_{2,h}$",
            title=r"Constant $h$",
        )

        axs[2].loglog(self.T / Ns, errs["hk"], ".-g", label="h=k")
        axs[2].loglog(
            self.T / Ns, (self.T / Ns) ** 2, "--", label=r"$\mathcal{O}{(h^2)}$"
        )
        axs[2].loglog(
            self.T / Ns, 0.005 * self.T / Ns, "--", label=r"$\mathcal{O}{(h)}$"
        )
        axs[2].set(
            xlabel=r"$h,k$",
            ylabel=r"$||E^n||_{2,h}$",
            title=r"$(h,k) \to (0,0)$ along $h=k$",
        )

        for i in (0, 1, 2):
            axs[i].legend()

        fig.tight_layout()
        fig.savefig(PLOTS_PATH / f"{filename}")


def plot3d(sol, x, t):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(*np.meshgrid(x, t), sol)
    ax.set(xlabel="x", ylabel="t")
    plt.show()


def task1():
    mu = 1 / 5
    a = 1
    b = 3 / 2 * np.pi
    phi = 1 / 4 * np.pi
    T = 1

    tester = ReacDiffTester(mu=mu, a=a, b=b, phi=phi, T=T)
    tester.generate_convergence_plots("task1_error.pdf")

    # plot3d(*tester.solver.solve(M=100, N=100, T=1))


#############################################
# ---------------- Task 2 -------------------
#############################################


def laplacian(A: np.ndarray) -> np.ndarray:
    return A[0:-2, 1:-1] + A[2:, 1:-1] + A[1:-1, 0:-2] + A[1:-1, 2:] - 4 * A[1:-1, 1:-1]


# Eq:
# St = −βIS + µ_S*∆S,
# It = βIS − γI + µ_I*∆I,
# Rt = γI


def SIR_solve(
    xy_end: float,
    t_end: float,
    N: int,
    M: int,
    S_0: np.ndarray,
    I_0: np.ndarray,
    R_0: np.ndarray,
    beta: Union[np.ndarray, float],
    gamma: float,
    mu_I: float,
    mu_S: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the differential eq. given by:\n
    St = -βIS + µ_S*∆S,
    It = βIS - γI + µ_I*∆I,
    Rt = γI

    (Uses forward Euler for the time step)
    ---
    Args:
        beta (Union[np.ndarray, float]): If array its size needs to be (N+2, N+2)

    Returns:

    """
    assert S_0.shape == (N, N), f"S_0 shape: {S_0.shape}. Expected: {(N, N)}"
    assert I_0.shape == (N, N), f"I_0 shape: {I_0.shape}. Expected: {(N, N)}"
    assert R_0.shape == (N, N), f"R_0 shape: {R_0.shape}. Expected: {(N, N)}"

    # t+[1], {S, I, R}, x+[4], y+[4]
    U = np.zeros((M + 1, 3, N + 4, N + 4), dtype=np.float64)
    U[0, 0, 2:-2, 2:-2] = S_0
    U[0, 1, 2:-2, 2:-2] = I_0
    U[0, 2, 2:-2, 2:-2] = R_0

    # Initial edge
    U[0, :, 0, :] = U[0, :, 1, :] = U[0, :, 2, :]
    U[0, :, -1, :] = U[0, :, -2, :] = U[0, :, -3, :]
    U[0, :, :, 0] = U[0, :, :, 1] = U[0, :, :, 2]
    U[0, :, :, -1] = U[0, :, :, -2] = U[0, :, :, -3]

    h = xy_end / N
    k = t_end / M

    for i in range(M):
        S, I, R = U[i, 0], U[i, 1], U[i, 2]
        # Inner points
        U[i + 1, 0, 1:-1, 1:-1] = (
            S[1:-1, 1:-1]
            - k * beta * I[1:-1, 1:-1] * S[1:-1, 1:-1]
            + k * mu_S / h**2 * laplacian(S)
        )
        U[i + 1, 1, 1:-1, 1:-1] = (
            I[1:-1, 1:-1]
            + k * beta * I[1:-1, 1:-1] * S[1:-1, 1:-1]
            - k * gamma * I[1:-1, 1:-1]
            + k * mu_I / h**2 * laplacian(I)
        )
        U[i + 1, 2, 1:-1, 1:-1] = R[1:-1, 1:-1] + k * gamma * I[1:-1, 1:-1]
        # Edge points
        U[i + 1, :, 0, :] = U[i + 1, :, 2, :]
        U[i + 1, :, -1, :] = U[i + 1, :, -3, :]
        U[i + 1, :, :, 0] = U[i + 1, :, :, 2]
        U[i + 1, :, :, -1] = U[i + 1, :, :, -3]

    return U, np.linspace(0, t_end, M + 1)


def plot_state(
    x: np.ndarray, y: np.ndarray, sol: np.ndarray, t, i_t: int, save: bool = False
) -> None:
    titles = ["Susceptible", "Infected", "Recovered"]

    for i in range(len(titles)):
        fig, ax = plt.subplots(subplot_kw={"xlabel": "x", "ylabel": "y"})
        im = ax.imshow(
            sol[i_t, i, :, :],
            extent=[0, 1, 0, 1],
            origin="lower",
        )
        fig.suptitle(rf"{titles[i]} at time {t[i_t]:.3f}")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(PLOTS_PATH / f"plot-i_t={i_t}-{i}.pdf")
        plt.close()
    return None


def population_deviation(u, t) -> None:
    initial = np.sum(u[0], axis=(0, 1, 2))
    people = np.sum(u, axis=(1, 2, 3))

    fig, ax = plt.subplots()
    ax.plot(t, (people - initial) / initial * 100)
    ax.set(
        xlabel="Time",
        ylabel="Population deviation (%)",
        title="Total population deviation"
    )
    ax.grid(True)
    plt.savefig(PLOTS_PATH / "pop_deviation.pdf")


def task2():
    N = 100
    M = 10_000
    a = 10

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

    S_0 = np.exp(-a / 2 * ((x_grid - 0.5) ** 2 / a + (y_grid - 0.25) ** 2))
    I_0 = np.exp(-a / 2 * ((x_grid - 0.75) ** 2 + (y_grid - 0.75) ** 2))
    R_0 = np.zeros((N, N))

    total_num_people = np.sum(S_0 + I_0 + R_0)

    S_0 /= total_num_people
    I_0 /= total_num_people
    R_0 /= total_num_people

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N + 2), np.linspace(0, 1, N + 2))
    beta = 100_000 * np.exp(-a / 2 * ((x_grid - 0.25) ** 2 + (y_grid - 0.5) ** 2 / a))
    gamma = 1
    mu_I = 1e-1
    mu_S = 1e-1

    u, t = SIR_solve(1, 1, N, M, S_0, I_0, R_0, beta, gamma, mu_I, mu_S)

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N + 4), np.linspace(0, 1, N + 4))
    for _i_t in (0, 100, 500, 1000, 2000, 3000, 4000, 5000, 10_000):
        plot_state(x_grid, y_grid, u, t, _i_t, save=True)

    population_deviation(u, t)


#############################################
# ---------------- Main ---------------------
#############################################


def main():
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    task1()
    task2()


if __name__ == "__main__":
    main()
