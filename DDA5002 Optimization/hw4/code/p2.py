import numpy as np
import matplotlib.pyplot as plt


# Define the function
def f(x1, x2):
    return x1**4 + 2 * (x1 - x2) * x1**2 + 4 * x2**2


# Generate grid points
x1 = np.linspace(-3, 1, 100)
x2 = np.linspace(-1, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Create contour plot
plt.figure(figsize=(8, 6))
contour = plt.contour(X1, X2, Z, levels=50, cmap="viridis")
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Contour plot of f(x)")
plt.colorbar(contour, label="f(x)")

# Mark the stationary points
plt.scatter(0, 0, color="red", s=50, label="Saddle point (0,0)")
plt.scatter(-2, 1, color="blue", s=50, label="Local minimizer (-2,1)")
plt.legend()
plt.grid(True)
plt.show()
