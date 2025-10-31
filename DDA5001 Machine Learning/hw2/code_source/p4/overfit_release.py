from itertools import product
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.cm as cm

np.random.seed(666)


def get_legendre_coe(K):
    """
    Input:
        K: the highest order polynomial degree
    Return:
        P: (K+1, K+1), the coefficient matrix, where i-th column corresponds
            to i-th legendre polynomial's coefficients.
    """
    # initialize the first two coefficients
    P = [np.array([1] + [0] * K), np.array([0, 1] + [0] * (K - 1))]
    for k in range(2, K + 1):
        P_k = np.zeros((K + 1,))
        P_k += (2 * k - 1) / k * np.roll(P[-1], 1) - (k - 1) / k * P[-2]
        P.append(P_k)

    return np.array(P).T


def generate_data(Qg, var, n):
    """
    Generate n data samples with Qg order legendre polynomial f and noise level var
    """
    x = np.random.uniform(-1, 1, (n,))
    epsilon = np.random.normal(0, np.sqrt(var), (n,))

    # get f
    normalize_factor = np.sqrt(np.sum([1 / (2 * q + 1) for q in range(Qg + 1)]))
    a = (
        np.random.normal(size=(Qg + 1,)) / normalize_factor
    )  # scale the variance of f to 1

    # get y
    Phi_x_Qg = np.vstack([np.power(x, i) for i in range(Qg + 1)]).T  # (n, Qg+1)
    P = get_legendre_coe(Qg)  # (Qg+1, Qg+1)
    y = Phi_x_Qg @ (P @ a) + epsilon

    return x, y, a


def legendre_matrix(x, K):
    """
    calculate the matrix of Legendre polynomial

    Input:
        x: (n,), the input data
        K: Integer, the highest order polynomial degree
    Return:
        A: (n, K+1), the matrix of Legendre polynomial
    """
    n = len(x)
    A = np.zeros((n, K + 1))
    A[:, 0] = 1.0
    if K >= 1:
        A[:, 1] = x
    for k in range(2, K + 1):
        A[:, k] = ((2 * k - 1) * x * A[:, k - 1] - (k - 1) * A[:, k - 2]) / k
    return A


def calBestFitCoefficients(x, y, K):
    """
    calculate the best degree-K legendre polynomial that fits data (x, y)

    Return:
        w_star: (K+1, ), the best-fit coefficients
    """
    A = legendre_matrix(x, K)
    w_star = np.linalg.inv(A.T @ A) @ A.T @ y
    return w_star


def calErout(w_star, a):
    """
    Input:
        w_star: (K+1, ), the best-fit coefficients
        a: (Qg+1, ), the true coefficients of the legendre polynomial
    Return:
        Erout: scalar, the out-of-sample error
    """
    K = len(w_star) - 1
    Qg = len(a) - 1
    max_order = max(K, Qg)
    Erout = 0.0

    for q in range(max_order + 1):
        if q <= K and q <= Qg:
            diff = w_star[q] - a[q]
        elif q <= K:
            diff = w_star[q]
        elif q <= Qg:
            diff = -a[q]
        else:
            diff = 0
        L_square = 1 / (2 * q + 1)
        Erout += (diff**2) * L_square

    return Erout


repeat_num = 40
ns = np.arange(20, 121, 5)
Qg_range = np.arange(1, 51)
vars = np.arange(0.0, 2.01, 0.05)

logs_exp1 = {
    "Erout_10": np.zeros((len(ns), len(Qg_range))),
    "Erout_2": np.zeros((len(ns), len(Qg_range))),
    "overfit_measure": np.zeros((len(ns), len(Qg_range))),
}

logs_exp2 = {
    "Erout_10": np.zeros((len(ns), len(vars))),
    "Erout_2": np.zeros((len(ns), len(vars))),
    "overfit_measure": np.zeros((len(ns), len(vars))),
}

var = 0.1
i, j = 0, 0
for Qg, n in product(Qg_range, ns):
    Erin_10, Erin_2 = 0, 0
    Erout_10, Erout_2 = 0, 0
    for _ in range(repeat_num):
        x, y, a = generate_data(Qg, var, n)
        w_star_2 = calBestFitCoefficients(x, y, 2)
        w_star_10 = calBestFitCoefficients(x, y, 10)
        Erout_2 += calErout(w_star_2, a) / repeat_num
        Erout_10 += calErout(w_star_10, a) / repeat_num
    overfit_measure = Erout_10 - Erout_2
    logs_exp1["Erout_2"][i, j] = Erout_2
    logs_exp1["Erout_10"][i, j] = Erout_10
    logs_exp1["overfit_measure"][i, j] = overfit_measure

    i += 1
    if i == len(ns):
        i = 0
        j += 1

Qg = 20
i, j = 0, 0
for var, n in product(vars, ns):
    Erin_10, Erin_2 = 0, 0
    Erout_10, Erout_2 = 0, 0
    for _ in range(repeat_num):
        x, y, a = generate_data(Qg, var, n)
        w_star_10 = calBestFitCoefficients(x, y, 10)
        w_star_2 = calBestFitCoefficients(x, y, 2)
        Erout_10 += calErout(w_star_10, a) / repeat_num
        Erout_2 += calErout(w_star_2, a) / repeat_num
    overfit_measure = Erout_10 - Erout_2
    logs_exp2["Erout_10"][i, j] = Erout_10
    logs_exp2["Erout_2"][i, j] = Erout_2
    logs_exp2["overfit_measure"][i, j] = overfit_measure

    i += 1
    if i == len(ns):
        i = 0
        j += 1

# clip for better plot view
for key in logs_exp1:
    logs_exp1[key] = np.clip(logs_exp1[key], -2, 10)
for key in logs_exp2:
    logs_exp2[key] = np.clip(logs_exp2[key], -2, 10)

# plot expriment 1
# cmap = cm.get_cmap("jet")
cmap = plt.get_cmap(
    "jet"
)  # use pyplot's get_cmap (recommended) to avoid MatplotlibDeprecationWarning (version 3.7+)
fig1, ax2 = plt.subplots(constrained_layout=True)
Qf_mesh, n_mesh = np.meshgrid(Qg_range, ns)
CS = ax2.contourf(n_mesh.T, Qf_mesh.T, logs_exp1["overfit_measure"].T, cmap=cmap)
ax2.set_title("Impact of $Q_g$ and $n$")
ax2.set_xlabel("Number of Data Points $n$")
ax2.set_ylabel("Noise level $Q_g$")
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel("Overfit Measure")
N = cmap.N
CS.cmap.set_under(cmap(1))
CS.cmap.set_over(cmap(N - 1))
# plt.savefig("overfit_Qg_vs_n.pdf")
plt.show()

# plot expriment 2
# cmap = cm.get_cmap("jet")
cmap = plt.get_cmap(
    "jet"
)  # use pyplot's get_cmap (recommended) to avoid MatplotlibDeprecationWarning (version 3.7+)
fig1, ax2 = plt.subplots(constrained_layout=True)
Qf_mesh, n_mesh = np.meshgrid(vars, ns)
CS = ax2.contourf(n_mesh.T, Qf_mesh.T, logs_exp2["overfit_measure"].T, cmap=cmap)
ax2.set_title("Impact of $\sigma$ and $n$")
ax2.set_xlabel("Number of Data Points $n$")
ax2.set_ylabel("Noise level $\sigma$")

cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel("Overfit Measure")
N = cmap.N
CS.cmap.set_under(cmap(1))
CS.cmap.set_over(cmap(N - 1))
# plt.savefig("overfit_sigma_vs_n.pdf")
plt.show()
