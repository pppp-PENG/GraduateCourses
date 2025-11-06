from sklearn.datasets import fetch_openml
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle

RND = 0
np.random.seed(RND)

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

# print(f"Dataset type: X={type(X)}, y={type(y)}") # pandas DataFrame/Series

# Plot jth image
# j = 0
# plt.title("The jth image is a {label}".format(label=int(y[j])))
# plt.imshow(X.loc[j].values.reshape((28, 28)), cmap="gray")
# plt.show()

if hasattr(X, "to_numpy"):
    X = X.to_numpy()
else:
    X = np.array(X)
y = np.array(y, dtype=int)

X = X.astype(np.float32) / 255.0

n_train_per_class = 300
n_val_per_class = 100
n_classes = 10

indices = np.arange(len(y))
train_idx = []
val_idx = []

for c in range(n_classes):
    cls_idx = indices[y == c]
    cls_idx = shuffle(cls_idx, random_state=RND)
    if len(cls_idx) < n_train_per_class + n_val_per_class:
        raise ValueError(f"Not enough samples for class {c}")
    train_idx.extend(cls_idx[:n_train_per_class])
    val_idx.extend(cls_idx[n_train_per_class : n_train_per_class + n_val_per_class])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
all_selected = set(train_idx.tolist() + val_idx.tolist())
test_idx = np.array([i for i in indices if i not in all_selected])

X_train = X[train_idx]
y_train = y[train_idx]
X_val = X[val_idx]
y_val = y[val_idx]
X_test = X[test_idx]
y_test = y[test_idx]

print("Sizes: train", X_train.shape, "val", X_val.shape, "test", X_test.shape)

out_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(out_dir, exist_ok=True)


def evaluate_C_grid(kernel_kwargs, C_grid, title_prefix):
    val_errors = []
    for C in C_grid:
        clf = svm.SVC(C=C, **kernel_kwargs)
        clf.fit(X_train, y_train)
        val_err = 1.0 - clf.score(X_val, y_val)
        val_errors.append(val_err)
        # print(f"{title_prefix} C={C:.4f} val_err={val_err:.4f}")
    val_errors = np.array(val_errors)
    # plot
    plt.figure()
    plt.semilogx(C_grid, val_errors, marker="o")
    plt.xlabel("C")
    plt.ylabel("validation error")
    plt.title(f"{title_prefix} validation error vs C")
    plt.grid(True)
    fname = os.path.join(out_dir, f"{title_prefix.replace(' ','_')}_valerr_vs_C.png")
    plt.savefig(fname, dpi=150)
    print("Saved plot:", fname)
    # pick best C
    best_idx = np.argmin(val_errors)
    best_C = C_grid[best_idx]
    print(f"{title_prefix} best C = {best_C:.4f}, val_err = {val_errors[best_idx]:.4f}")
    # retrain on train+val
    # X_tr_full = np.vstack([X_train, X_val])
    # y_tr_full = np.hstack([y_train, y_val])
    X_tr_full = X
    y_tr_full = y
    clf_best = svm.SVC(C=best_C, **kernel_kwargs)
    clf_best.fit(X_tr_full, y_tr_full)
    test_err = 1.0 - clf_best.score(X_test, y_test)
    print(f"{title_prefix} test error (retrained on train+val) = {test_err:.4f}")
    return best_C, val_errors, test_err


# (a)
C_grid = np.logspace(-3, 3, 10)

print("\nEvaluating poly degree=1 (inhomogeneous linear)...")
poly1_kwargs = dict(kernel="poly", degree=1, coef0=1.0, gamma="auto")
best_C_poly1, val_errs_poly1, test_err_poly1 = evaluate_C_grid(
    poly1_kwargs, C_grid, "poly_deg1"
)

print("\nEvaluating poly degree=2 (quadratic)...")
poly2_kwargs = dict(kernel="poly", degree=2, coef0=1.0, gamma="auto")
best_C_poly2, val_errs_poly2, test_err_poly2 = evaluate_C_grid(
    poly2_kwargs, C_grid, "poly_deg2"
)

# (b)
print("\nEvaluating RBF grid search over C and gamma...")
gamma_grid = np.logspace(-4, 1, 6)
val_errors_rbf = np.zeros((len(gamma_grid), len(C_grid)))
for i, gamma in enumerate(gamma_grid):
    for j, C in enumerate(C_grid):
        clf = svm.SVC(C=C, kernel="rbf", gamma=gamma)
        clf.fit(X_train, y_train)
        val_errors_rbf[i, j] = 1.0 - clf.score(X_val, y_val)
        # print(f"rbf gamma={gamma:.1e} C={C:.4f} val_err={val_errors_rbf[i,j]:.4f}")

plt.figure(figsize=(8, 5))
for i, gamma in enumerate(gamma_grid):
    plt.semilogx(C_grid, val_errors_rbf[i, :], marker="o", label=f"gamma={gamma:.0e}")
plt.xlabel("C")
plt.ylabel("validation error")
plt.title("RBF validation error vs C (one line per gamma)")
plt.grid(True)
plt.xticks(C_grid, [f"{c:.0e}" for c in C_grid], rotation=45)
plt.legend(title="gamma", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
fname = os.path.join(out_dir, "rbf_valerr_vs_C_per_gamma.png")
plt.savefig(fname, dpi=150)
print("Saved RBF val-vs-C plot:", fname)

best_idx = np.unravel_index(np.argmin(val_errors_rbf), val_errors_rbf.shape)
best_gamma = gamma_grid[best_idx[0]]
best_C_rbf = C_grid[best_idx[1]]
best_val_rbf = val_errors_rbf[best_idx]
print(
    f"RBF best gamma={best_gamma:.4f}, best C={best_C_rbf:.4f}, val_err={best_val_rbf:.4f}"
)

# X_tr_full = np.vstack([X_train, X_val])
# y_tr_full = np.hstack([y_train, y_val])
X_tr_full = X
y_tr_full = y
clf_rbf_best = svm.SVC(C=best_C_rbf, kernel="rbf", gamma=best_gamma)
clf_rbf_best.fit(X_tr_full, y_tr_full)
test_err_rbf = 1.0 - clf_rbf_best.score(X_test, y_test)
print(f"RBF test error (retrained on train+val) = {test_err_rbf:.4f}")

summary = {
    "poly_deg1": {"best_C": best_C_poly1, "test_err": test_err_poly1},
    "poly_deg2": {"best_C": best_C_poly2, "test_err": test_err_poly2},
    "rbf": {"best_C": best_C_rbf, "best_gamma": best_gamma, "test_err": test_err_rbf},
}
print("\nSummary:", summary)
