import scipy as sc
import matplotlib.pyplot as plt
import numpy as np

# Eq:
# St = −βIS + µ_S*∆S,
# It = βIS − γI + µ_I*∆I,
# Rt = γI


class Solver:
    """Solve the differential eq. given by:\n
    St = −βIS + µ_S*∆S,
    It = βIS − γI + µ_I*∆I,
    Rt = γI
    """

    def __init__(
        self,
        N: int,
        M: int,
        S_0: np.ndarray,
        I_0: np.ndarray,
        R_0: np.ndarray,
        beta: float,
        gamma: float,
        mu_I: float,
        mu_S: float,
    ) -> None:
        assert S_0.shape == (N, N), f"S_0 shape: {S_0.shape}. Expected: {(N, N)}"
        assert I_0.shape == (N, N), f"I_0 shape: {I_0.shape}. Expected: {(N, N)}"
        assert R_0.shape == (N, N), f"R_0 shape: {R_0.shape}. Expected: {(N, N)}"

        self.N = N
        self.M = M

        self.S_0 = S_0
        self.I_0 = I_0
        self.R_0 = R_0

        # t+[1], {S, I, R}, x+[2], y+[2]
        self.U = np.zeros((M + 1, 3, N + 2, N + 2), dtype=np.float64)
        self.U[0, 0, 1:-1, 1:-1] = S_0
        self.U[0, 1, 1:-1, 1:-1] = I_0
        self.U[0, 2, 1:-1, 1:-1] = R_0

        self.beta = beta
        self.gamma = gamma
        self.mu_I = mu_I
        self.mu_S = mu_S

    def solve(
        self,
        xy_end: float,
        t_end: float,
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
        h = xy_end / self.N
        k = t_end / self.M

        for i in range(self.M):
            S, I, R = self.U[i, 0], self.U[i, 1], self.U[i, 2]
            self.U[i + 1, 0, 1:-1, 1:-1] = (
                S[1:-1, 1:-1]
                - k * self.beta * I[1:-1, 1:-1] * S[1:-1, 1:-1]
                + k * self.mu_S / h**2 * self.laplacian(S)
            )
            self.U[i + 1, 1, 1:-1, 1:-1] = (
                I[1:-1, 1:-1]
                + k * self.beta * I[1:-1, 1:-1] * S[1:-1, 1:-1]
                - k * self.gamma * I[1:-1, 1:-1]
                + k * self.mu_I / h**2 * self.laplacian(I)
            )
            self.U[i + 1, 2, 1:-1, 1:-1] = (
                R[1:-1, 1:-1] + k * self.gamma * I[1:-1, 1:-1]
            )

        return self.U

    def laplacian(self, A: np.ndarray) -> np.ndarray:
        return (
            A[0:-2, 1:-1]
            + A[2:, 1:-1]
            + A[1:-1, 0:-2]
            + A[1:-1, 2:]
            - 4 * A[1:-1, 1:-1]
        )

if __name__ == "__main__":
    N = 100
    M = 1000
    S_0 = np.ones((N, N))*100
    I_0 = np.zeros((N, N))
    I_0[0, 0] = 10
    R_0 = np.zeros((N, N))
    beta = 1e-2
    gamma = 1e-4
    mu_I = 1e-4
    mu_S = 1e-4

    s = Solver(N, M, S_0, I_0, R_0, beta, gamma, mu_I, mu_S)
    u = s.solve(1, 50)
    # print(u)
    plt.imshow(u[0, 1])
    plt.show()
    plt.imshow(u[500, 1])
    plt.show()
    plt.imshow(u[1000, 1])
    plt.show()
