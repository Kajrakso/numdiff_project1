import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

beta = 3
gma = 1

def fun(x, y):
    S, I, R = y[0], y[1], y[2]
    return np.array([
        -beta * S * I,
        beta * S * I - gma * I,
        gma * I
    ])

def main():
    N = 100
    tinit = 0
    tend = 10
    teval = np.linspace(tinit, tend, N)

    # initial fraction of population that is infected
    alpha = 0.1
    yinit = np.array([1 - alpha, alpha, 0])

    sol = solve_ivp(fun, t_span=(tinit, tend), y0=yinit, t_eval=teval)

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label="S")
    ax.plot(sol.t, sol.y[1], label="I")
    ax.plot(sol.t, sol.y[2], label="R")
    ax.set(title="", xlabel="time", ylabel="population fraction")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
