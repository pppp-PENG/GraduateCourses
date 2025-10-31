import os
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(666)


def load_data_img(path, classes, img_size=32):
    """load image dataset
    Input:
        path: [str] path of dir which contains the subfolders of different classes' images
        classes: [list] class names
    Return:
        X: (n, d), data matrix
        Y: (n, ), corresponding label
    """
    if os.path.exists(path + "X.npy"):
        X = np.load(path + "X.npy")
        Y = np.load(path + "Y.npy")
        return X, Y

    X, Y = [], []
    for y, cls in enumerate(classes):
        data_path = Path(path + cls)
        for p in data_path.iterdir():
            img = ImageOps.grayscale(Image.open(f"{p}"))
            img = img.resize((img_size, img_size))
            x = np.array(img).flatten()
            X.append(x)
            Y.append(y)

    X, Y = np.array(X), np.array(Y)
    np.save(path + "X.npy", X)
    np.save(path + "Y.npy", Y)

    return X, Y


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cal_loss(pred, Y_onehot):
    """calculate multinomial logistic regression loss
    Input:
        pred: (n, K), softmax output
        Y_onehot: (n, K), onehot label
    Output:
        logistic loss
    """
    return -np.mean(np.log(np.sum(np.multiply(pred, Y_onehot), axis=1)))


# ============ code for loading coffee dataset ============
# classes = ["Dark", "Green", "Light", "Medium"]
# X, Y = load_data_img("coffee/train/", classes)
# X_test, Y_test = load_data_img("coffee/test/", classes)
# n, n_test = X.shape[0], X_test.shape[0]
# mu = 1e-2


# ============ code for loading weather dataset ============
classes = os.listdir("weather/dataset")
X_all, Y_all = load_data_img("weather/dataset/", classes)
data_num = X_all.shape[0]
train_idx = np.random.choice(data_num, size=int(data_num * 0.8), replace=False)
test_idx = np.delete(np.arange(data_num), train_idx)
X, Y = X_all[train_idx], Y_all[train_idx]
X_test, Y_test = X_all[test_idx], Y_all[test_idx]
n, n_test = X.shape[0], X_test.shape[0]
mu = 2e-2


# normalize data
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std
X_test = (X_test - mean) / std

inlcude_bias = True
optimizers = ["agd", "gd"]

if inlcude_bias:
    X = np.concatenate([X, np.ones(shape=(n, 1))], axis=1)
    X_test = np.concatenate([X_test, np.ones(shape=(n_test, 1))], axis=1)

d = X.shape[1]
K = np.max(Y) + 1
Y_onehot = np.eye(K)[Y]  # (n, K)
Y_test_onehot = np.eye(K)[Y_test]  # (n', K)

# hyperparameters
epochs = 1000

# initialization
Theta = np.zeros(shape=(d, K))
hist = {
    "agd": {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []},
    "gd": {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []},
}

# train
for opt in optimizers:
    Theta = np.zeros(shape=(d, K))

    for epoch in range(epochs):

        # evaluate
        if epoch % 5 == 0:
            pred_train = softmax(np.matmul(X, Theta))  # (n, K)
            train_loss = cal_loss(pred_train, Y_onehot)
            pred_test = softmax(np.matmul(X_test, Theta))  # (n', K)
            test_loss = cal_loss(pred_test, Y_test_onehot)

            # print(pred_train)
            # print(train_loss)
            # print(pred_test)
            # print(test_loss)
            # quit()

            test_acc = np.sum(pred_test.argmax(axis=1) == Y_test) / n_test
            train_acc = np.sum(pred_train.argmax(axis=1) == Y) / n

            hist[opt]["train_loss"].append(train_loss)
            hist[opt]["test_loss"].append(test_loss)
            hist[opt]["train_acc"].append(train_acc)
            hist[opt]["test_acc"].append(test_acc)

            # print(
            #     f"epoch:{epoch}, train_loss:{train_loss:.5f}, test_loss:{test_loss:.5f}, test_acc:{test_acc:.4f}, train_acc:{train_acc:.4f}"
            # )

        n_samples = X.shape[0]

        # Accelerated gradient descent
        if opt == "agd":
            if epoch == 0:
                grad = (
                    np.matmul(X.T, (softmax(np.matmul(X, Theta)) - Y_onehot))
                    / n_samples
                )
                Theta = Theta - mu * grad
                Theta_prev = Theta.copy()
            else:
                k = epoch + 1
                w = Theta + ((k - 1) / (k + 2)) * (Theta - Theta_prev)
                grad = np.matmul(X.T, (softmax(np.matmul(X, w)) - Y_onehot)) / n_samples
                w -= mu * grad
                Theta_prev = Theta.copy()
                Theta = w

        # Gradient descent
        elif opt == "gd":
            grad = np.matmul(X.T, (softmax(np.matmul(X, Theta)) - Y_onehot)) / n_samples
            Theta -= mu * grad


hist_idx = np.arange(epochs / 5)

# TODO plot the training loss and testing loss of AGD and GD in the SAME figure.
plt.figure(figsize=(10, 6))
plt.plot(
    hist_idx * 5,
    hist["agd"]["train_loss"],
    color="blue",
    linestyle="-",
    label="AGD Train Loss",
    linewidth=1,
)
plt.plot(
    hist_idx * 5,
    hist["agd"]["test_loss"],
    color="blue",
    linestyle="--",
    label="AGD Test Loss",
    linewidth=1,
)
plt.plot(
    hist_idx * 5,
    hist["gd"]["train_loss"],
    color="orange",
    linestyle="-",
    label="GD Train Loss",
    linewidth=1,
)
plt.plot(
    hist_idx * 5,
    hist["gd"]["test_loss"],
    color="orange",
    linestyle="--",
    label="GD Test Loss",
    linewidth=1,
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Test Loss: AGD vs GD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# TODO plot the training accuracy and testing accuracy of AGD and GD in the SAME figure.
plt.figure(figsize=(10, 6))
plt.plot(
    hist_idx * 5,
    hist["agd"]["train_acc"],
    color="blue",
    linestyle="-",
    label="AGD Train Acc",
    linewidth=1,
)
plt.plot(
    hist_idx * 5,
    hist["agd"]["test_acc"],
    color="blue",
    linestyle="--",
    label="AGD Test Acc",
    linewidth=1,
)
plt.plot(
    hist_idx * 5,
    hist["gd"]["train_acc"],
    color="orange",
    linestyle="-",
    label="GD Train Acc",
    linewidth=1,
)
plt.plot(
    hist_idx * 5,
    hist["gd"]["test_acc"],
    color="orange",
    linestyle="--",
    label="GD Test Acc",
    linewidth=1,
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train/Test Accuracy: AGD vs GD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
