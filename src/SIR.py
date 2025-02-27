import scipy as sc
import matplotlib.pyplot as plt
import numpy as np

# Eq:
# St = −βIS + µ_S*∆S,
# It = βIS − γI + µ_I*∆I,
# Rt = γI


def solve(
    xy_end: float,
    t_end: float,
    N: int,
    M: int,
    S_0: np.ndarray,
    I_0: np.ndarray,
    R_0: np.ndarray,
    beta: float,
    gamma: float,
    mu_I: float,
    mu_S: float,
) -> np.ndarray:
    """Solve the differential eq. given by:\n
    St = −βIS + µ_S*∆S,
    It = βIS − γI + µ_I*∆I,
    Rt = γI

    (Uses forward Euler for the time step)
    ---
    Args:

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

    return U


def laplacian(A: np.ndarray) -> np.ndarray:
    return A[0:-2, 1:-1] + A[2:, 1:-1] + A[1:-1, 0:-2] + A[1:-1, 2:] - 4 * A[1:-1, 1:-1]


if __name__ == "__main__":
    N = 100
    M = 10_000
    S_0 = np.ones((N, N)) * 100
    I_0 = np.zeros((N, N))
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    I_0[:, :] = np.sin(np.pi / 2 * (x_grid.T + y_grid))
    R_0 = np.zeros((N, N))
    beta = 10
    gamma = 10
    mu_I = 1e-2
    mu_S = 1e-2

    u = solve(1, 1, N, M, S_0, I_0, R_0, beta, gamma, mu_I, mu_S)
    # print(u)
    plt.plot(u[:, 0, 2, 2], label="S")
    plt.plot(u[:, 1, 2, 2], label="I")
    plt.plot(u[:, 2, 2, 2], label="R")
    plt.legend()
    print("xy=2:\n", u[:, 0, 2, 2] + u[:, 1, 2, 2] + u[:, 2, 2, 2], "\n")
    print("xy=50:\n", u[:, 0, 50, 50] + u[:, 1, 50, 50] + u[:, 2, 50, 50], "\n")
    plt.show()
    plt.cla()

    # Plot total inhabitants
    plt.plot(np.sum(u, axis=(1, 2, 3)))
    print(np.sum(u, axis=(1, 2, 3)))
    plt.show()
    plt.cla()
    #
    # plt.imshow(u[0, 1])
    # # plt.imshow(np.sum(u[500], axis=0))
    # print(np.sum(u[0]))
    # plt.show()
    # plt.cla()
    #
    # plt.imshow(u[5000, 1])
    # # plt.imshow(np.sum(u[500], axis=0))
    # print(np.sum(u[500]))
    # plt.show()
    # plt.cla()
    #
    # plt.imshow(u[10_000, 1])
    # # plt.imshow(np.sum(u[10_000], axis=0))
    # print(np.sum(u[10_000]))
    # plt.show()
