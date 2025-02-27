import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


def plot_state(x: np.ndarray, y: np.ndarray, sol: np.ndarray, i_t: int) -> None:
    fig, axs = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
    for ax in axs:
        ax.set_zlim(0, 200)
    for i in range(len(axs)):
        axs[i].plot_surface(x, y, sol[i_t, i, :, :])
        axs[i].set_title("Recovered")
    fig.suptitle("State")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    N = 100
    M = 10000
    S_0 = np.ones((N, N)) * 100
    I_0 = np.zeros((N, N))
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    I_0[:, :] = 10 + 10 * np.sin(3 * np.pi * x_grid) * np.sin(3 * np.pi * y_grid)
    R_0 = np.zeros((N, N))
    beta = 1
    gamma = 10
    mu_I = 5e-2
    mu_S = 1e-2

    u = solve(1, 1, N, M, S_0, I_0, R_0, beta, gamma, mu_I, mu_S)

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N + 4), np.linspace(0, 1, N + 4))
    plot_state(x_grid, y_grid, u, 500)
    # anim = PlotAnimation(x_grid, y_grid, u, frames=500, interval=5)
    # a = anim.animate()
    # plt.show()
    # writer = animation.PillowWriter(fps=60)
    # a.save("scatter.gif", writer=writer)

    # fig = plt.figure()
    # axs = fig.subplots(1, 3, subplot_kw={"projection": "3d"})
    #
    # def update(frame):
    #     for i, ax in enumerate(axs):
    #         ax.clear()
    #         ax.set_zlim(0, 200)  # Adjust based on your data range
    #         ax.set_title(f"Plot {i+1}, Frame {frame}")
    #         ax.plot_surface(
    #             x_grid, y_grid, u[frame, i, :, :], cmap="viridis"
    #         )
    #     return axs
    #
    # # Initialize surface plots
    # for ax in axs:
    #     surf = ax.plot_surface(x_grid, y_grid, u[0, 0], cmap="viridis")
    #
    # ani = animation.FuncAnimation(fig, update, frames=10_000, interval=1)
    # plt.show()
