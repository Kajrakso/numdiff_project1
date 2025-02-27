import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

from SIR import solve


class PlotAnimation:
    def __init__(
        self, x: np.ndarray, y: np.ndarray, sol: np.ndarray, frames: int, interval: int
    ) -> None:
        self.x = x
        self.y = y
        self.sol = sol

        self.frames = frames
        self.index = np.linspace(0, self.sol.shape[0] - 1, self.frames, dtype=np.int64)
        self.interval = interval

        self.fig, self.axs = plt.subplots(1, 3, subplot_kw={"projection": "3d"})

        for ax in self.axs:
            ax.set_zlim(0, 200)

    def update(self, frame):
        for i in range(len(self.axs)):
            self.axs[i].clear()
            self.axs[i].set_zlim(0, 200)
            self.axs[i].plot_surface(
                self.x, self.y, self.sol[self.index[frame], i, :, :]
            )
        return self.axs

    def animate(self):
        for i in range(len(self.axs)):
            self.axs[i].plot_surface(self.x, self.y, self.sol[0, i, :, :])
        return animation.FuncAnimation(
            self.fig, self.update, frames=self.frames, interval=self.interval
        )


def plot_state(
    x: np.ndarray, y: np.ndarray, sol: np.ndarray, t, i_t: int, save: bool = False
) -> None:
    titles = ["Susceptible", "Infected", "Recovered"]
    cmaps = ["Reds", "YlGn", "Blues"]

    plt.rcParams["font.size"] = 12
    fig, axs = plt.subplots(
        1,
        3,
        subplot_kw={"projection": "3d", "xlabel": "x", "ylabel": "y"},
        figsize=(20, 6),
    )
    for ax in axs:
        ax.set_zlim(0, np.max(sol[i_t]) * 1.02)
    for i in range(len(axs)):
        axs[i].plot_surface(
            x,
            y,
            sol[i_t, i, :, :],
            cmap=cmaps[i],
            vmin=np.max(np.min(sol[i_t, i] - 10), 0),
        )
        axs[i].set_title(titles[i])
        axs[i].set_zlabel(titles[i])
    fig.suptitle(f"Plot of the SIR-model at time {t[i_t]:.3f}", fontsize=16)
    fig.tight_layout()
    if save:
        plt.savefig(f"./Images/plot-i_t={i_t}.png")
        plt.savefig(f"./Images/plot-i_t={i_t}.svg")
    else:
        plt.show()


if __name__ == "__main__":
    N = 100
    M = 10000
    S_0 = np.ones((N, N)) * 100
    I_0 = np.zeros((N, N))
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    I_0[:, :] = 10 * (1 + np.sin(3 * np.pi * x_grid) * np.sin(3 * np.pi * y_grid))
    R_0 = np.zeros((N, N))
    beta = 1
    gamma = 10
    mu_I = 5e-2
    mu_S = 1e-2

    u, t = solve(1, 1, N, M, S_0, I_0, R_0, beta, gamma, mu_I, mu_S)

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N + 4), np.linspace(0, 1, N + 4))
    plot_state(x_grid, y_grid, u, t, 500, save=True)
