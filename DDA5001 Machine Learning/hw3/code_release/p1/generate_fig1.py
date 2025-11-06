import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )
except Exception as e:
    pass

df = pd.read_csv("training_data.csv")
x_data = df["x"]
y_data = df["y"]

xx = np.arange(-1.5, 1.55, 0.05)
yy = xx**2

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(
    x_data,
    y_data,
    label="data",
    facecolors="none",
    edgecolors="#2E8CC9",
    linewidths=1.5,
)
ax.plot(xx, yy, linewidth=2, color="#D95319", label="target function")

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-0.5, 2.5])
ax.set_xticks([-1, 0, 1])

legend_font_properties = {"family": "Times New Roman", "size": 20}
ax.legend(prop=legend_font_properties, edgecolor="black")

fig.set_facecolor("white")

ax.spines["top"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
ax.spines["left"].set_linewidth(1.5)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontsize(20)

ax.tick_params(direction="in", width=1.5, length=6)

ax.set_xlabel("$x$", fontsize=25, fontname="Times New Roman")
ax.set_ylabel("$y$", fontsize=25, fontname="Times New Roman")

plt.tight_layout()

# To save the figure, uncomment the line below:
# fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight')

from sklearn.model_selection import KFold


def poly_design(x, degree=8):
    x = np.asarray(x).reshape(-1)
    return np.vander(x, N=degree + 1, increasing=True)


df_test = pd.read_csv("test_data.csv")
x_test = df_test["x_test"].values
y_test = df_test["y_test"].values

X_train = poly_design(x_data.values, degree=8)
y_train = y_data.values

theta_ls, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
yy_fit = np.dot(poly_design(xx, degree=8), theta_ls)
ax.plot(xx, yy_fit, linewidth=2, color="#4DBF4A", label="8th poly fit (LS)")
ax.legend(prop=legend_font_properties, edgecolor="black")
print("a2: theta (LS):", theta_ls)

X_test = poly_design(x_test, degree=8)
res_test_ls = X_test.dot(theta_ls) - y_test
test_error_ls = np.linalg.norm(res_test_ls)
print("a3: test error ||X_test theta_ls - y_test||_2 = {:.4f}".format(test_error_ls))

lambda_candidates = np.array(
    [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 0.5, 0.8, 1, 2, 5, 10, 15, 20, 50, 100],
    dtype=float,
)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
val_errors = []

for lam in lambda_candidates:
    fold_errors = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        A = X_tr.T.dot(X_tr) + lam * np.eye(X_tr.shape[1])
        theta_ridge = np.linalg.solve(A, X_tr.T.dot(y_tr))
        y_val_pred = X_val.dot(theta_ridge)
        mse = np.linalg.norm(y_val_pred - y_val, ord=2)
        fold_errors.append(mse)
    val_errors.append(np.mean(fold_errors))

val_errors = np.array(val_errors)
best_idx = np.argmin(val_errors)
best_lambda = lambda_candidates[best_idx]
print("b1: best lambda (5-fold CV, by avg MSE) = {}".format(best_lambda))

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(lambda_candidates, val_errors, marker="o")
ax2.set_xscale("log")
ax2.set_xlabel("lambda (log scale)", fontsize=12)
ax2.set_ylabel("validation MSE", fontsize=12)
ax2.set_title("Validation error vs lambda (5-fold CV)")
ax2.grid(True, which="both", ls="--", linewidth=0.5)

fig2.tight_layout()
fig2.show()

selected_lambdas = [0.01, 0.1, 0.8, 5.0]
test_errors_selected = {}

for lam in selected_lambdas:
    A = X_train.T.dot(X_train) + lam * np.eye(X_train.shape[1])
    theta_sel = np.linalg.solve(A, X_train.T.dot(y_train))
    fig_sel, ax_sel = plt.subplots(figsize=(6, 5))
    ax_sel.scatter(
        x_data,
        y_data,
        label="data",
        facecolors="none",
        edgecolors="#2E8CC9",
        linewidths=1.5,
    )
    ax_sel.plot(xx, yy, linewidth=2, color="#D95319", label="target function")
    yy_sel = np.dot(poly_design(xx, degree=8), theta_sel)
    ax_sel.plot(xx, yy_sel, linewidth=2, color="#8B008B", label=f"lambda={lam}")
    ax_sel.set_xlim([-1.5, 1.5])
    ax_sel.set_ylim([-0.5, 2.5])
    ax_sel.set_xticks([-1, 0, 1])
    ax_sel.legend()
    ax_sel.set_xlabel("$x$")
    ax_sel.set_ylabel("$y$")
    fig_sel.tight_layout()
    fig_sel.show()

    res = X_test.dot(theta_sel) - y_test
    err = np.linalg.norm(res)
    test_errors_selected[lam] = err
    print(
        "b2/b3: lambda = {:g}, theta = {}, test L2 error = {:.4f}".format(
            lam, theta_sel, err
        )
    )


plt.show()
