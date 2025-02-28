import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path

PLOTS_PATH = Path("./Plots/")

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

    return U, np.linspace(0, t_end, M+1)

def plot_state(
    x: np.ndarray, y: np.ndarray, sol: np.ndarray, t, i_t: int, save: bool = False
) -> None:
    titles = ["Susceptible", "Infected", "Recovered"]

    for i in range(len(titles)):
        fig, ax = plt.subplots(
            subplot_kw={"xlabel": "x", "ylabel": "y"}
        )
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

    plt.plot(t, (people - initial) / initial * 100)
    plt.xlabel("Time")
    plt.ylabel("Population deviation (%)")
    plt.grid()
    plt.title("Total population deviation")
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

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N+2), np.linspace(0, 1, N+2))
    beta = 100_000*np.exp(-a / 2 * ((x_grid - 0.25) ** 2 + (y_grid - 0.5)**2 / a))
    gamma = 1
    mu_I = 1e-1
    mu_S = 1e-1

    u, t = SIR_solve(1, 1, N, M, S_0, I_0, R_0, beta, gamma, mu_I, mu_S)

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N + 4), np.linspace(0, 1, N + 4))
    for _i_t in (0, 100, 500, 1000, 2000, 3000, 4000, 5000, 10_000):
        plot_state(x_grid, y_grid, u, t, _i_t, save=True)

    population_deviation(u, t)


def main():
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    task2()


if __name__ == "__main__":
    main()
