from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


def solver(
    fx0: Callable,
    fx1: Callable,
    ft0: Callable,
    t_end: float,
    mu: float,
    a: float,
    N: int,
    M: int,
) -> np.ndarray:
    h = 1 / N
    k = 1 / M
    # The things we solve for has shape (x, t) in (N, M).
    # The x's we solve for are in the range [1, N], and conditions on {0, N+1}
    # The t's we solve for are in the range [1, M], and conditions on {0}
    U = np.zeros((N + 2, M + 1))
    U[0, :] = fx0(np.linspace(0, t_end, M + 1))
    U[N + 1, :] = fx1(np.linspace(0, t_end, M + 1))
    U[:, 0] = ft0(np.linspace(0, 1, N + 2))
    r = mu * k / h**2

    for i in range(N):
        A = sc.sparse.diags_array(
            [-r / 2, 1 + r / 2, -r / 2], offsets=[-1, 0, 1], shape=(N, N)
        )
        rhs = U[1:-1, i] * (1 + k * a) + r / 2 * (U[2:, i] - 2 * U[1:-1, i] + U[:-2, i])
        U_star = sc.sparse.linalg.spsolve(A, rhs)
        U[1:-1, i+1] = U_star + k*a/2*(U_star - U[1:-1, i])

    return U


def fx0(t: np.ndarray) -> np.ndarray:
    return 0*t

def fx1(t: np.ndarray) -> np.ndarray:
    return np.ones(t.shape)

def ft0(x: np.ndarray) -> np.ndarray:
    return x

if __name__ == "__main__":
    mu = 1
    a = 2
    N = 1000
    M = 1000
    t_end = 10
    U = solver(fx0, fx1, ft0, t_end, mu, a, N, M)
    print(U)
    
    x = np.linspace(0, 1, N+2)
    plt.plot(x, U[:, 0])
    plt.show()
