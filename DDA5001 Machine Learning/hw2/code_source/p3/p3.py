import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)

script_dir = os.path.dirname(os.path.abspath(__file__))

A = np.load(os.path.join(script_dir, "A.npy"))  # (500, 30)
b = np.load(os.path.join(script_dir, "b.npy"))  # (500, )
x_star = np.load(os.path.join(script_dir, "x_star.npy"))  # (30, )

epochs = 300
n, d = A.shape
mu_constant = 0.005
mu_poly_initial = 0.03
mu_geo_initial = 0.01
geo_decay = 0.9
optimizers = ["gd_const", "gd_poly", "gd_geo"]
optimality_gap = {"gd_const": [], "gd_poly": [], "gd_geo": []}


def subgradient(x):
    Ax_b = A @ x - b
    sign_vec = np.sign(Ax_b)
    zero_mask = Ax_b == 0
    sign_vec[zero_mask] = np.random.choice([-1, 1], size=np.sum(zero_mask))
    return A.T @ sign_vec


for opt in optimizers:
    x = np.zeros(d)
    for epoch in range(epochs):
        grad = subgradient(x)
        if opt == "gd_const":
            mu = mu_constant
        elif opt == "gd_poly":
            mu = mu_poly_initial / np.sqrt(epoch + 1)
        elif opt == "gd_geo":
            mu = mu_geo_initial * (geo_decay ** (epoch + 1))
        x = x - mu * grad
        gap = np.linalg.norm(x - x_star, ord=2)
        optimality_gap[opt].append(gap)

epochs_plot = np.arange(epochs)
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for opt in optimizers:
    plt.plot(epochs_plot, optimality_gap[opt], label=opt, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Optimality Gap ||x_k - x*||_2")
plt.title("Optimality Gap (Normal Scale)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for opt in optimizers:
    plt.semilogy(epochs_plot, optimality_gap[opt], label=opt, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Optimality Gap ||x_k - x*||_2 (log scale)")
plt.title("Optimality Gap (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
