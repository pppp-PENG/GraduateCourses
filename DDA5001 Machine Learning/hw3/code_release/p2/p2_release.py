from os.path import join
from PIL.Image import open
import matplotlib.pyplot as plt
import numpy as np


def preprocess(V):
    """
    Preprocess ORL faces data matrix as Stan Li, et. al.

    Return normalized and preprocessed data matrix.

    :param V: The ORL faces data matrix.
    :type V: `numpy.matrix`
    """
    print("Data preprocessing")
    min_val = V.min(axis=0)
    V = V - np.asmatrix(np.ones((V.shape[0], 1))) * min_val
    max_val = V.max(axis=0) + 1e-4
    V = (255.0 * V) / (np.asmatrix(np.ones((V.shape[0], 1))) * max_val) / 100.0
    return V


def read_orl(img_size=(112, 92)):
    """
    Read face image data from the ORL database. The matrix's shape is 2576 (pixels) x 400 (faces).

    Step through each subject and each image. Reduce the size of the images by a factor of 0.5.

    Return the ORL faces data matrix.
    """
    print("Reading ORL faces database")
    dir = join("ORL_faces", "s")
    V = np.matrix(np.zeros((img_size[0] * img_size[1], 400)))
    for subject in range(40):
        for image in range(10):
            im = open(join(dir + str(subject + 1), str(image + 1) + ".pgm"))
            # reduce the size of the image
            im = im.resize(img_size[::-1])
            V[:, 10 * subject + image] = np.asmatrix(np.asarray(im).flatten()).T
    return V


def cal_snr(img_origin, img_recon):
    ratio = np.sum(np.power(img_recon, 2)) / np.sum(
        np.power((img_origin - img_recon), 2)
    )
    return 10 * np.log10(ratio)


img_size = (112, 92)
X = read_orl(img_size)
X_processed = preprocess(X)

# plot the eigenfaces, you may use plt.imshow(X[:, i].reshape(*img_size), cmap='gray') to visualize the data

# plot original img & reconstructed img

# calculate snr

X_arr = np.asarray(X_processed, dtype=float)
d, n = X_arr.shape

mean_vec = np.mean(X_arr, axis=1, keepdims=True)
Xc = X_arr - mean_vec

U, S, VT = np.linalg.svd(Xc, full_matrices=False)

top_k = 40
eigenfaces = U[:, :top_k]

fig_ef, axes = plt.subplots(5, 8, figsize=(12, 8))
fig_ef.suptitle("Top 40 Eigenfaces", fontsize=16)
for i, ax in enumerate(axes.flat):
    ef = eigenfaces[:, i].reshape(img_size)
    vmax = np.abs(ef).max()
    ax.imshow(ef, cmap="gray", vmin=-vmax, vmax=vmax)
    ax.axis("off")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_ef.show()

sample_indices = [0, 50, 100, 150, 200]
ks = [20, 40, 100, 200, 300]

coeffs_full = U.T.dot(Xc)

fig_rec, axes_rec = plt.subplots(len(sample_indices), len(ks) + 1, figsize=(12, 10))
fig_rec.suptitle(
    "Original and Reconstructions (rows: images; cols: original, k=20,40,100,200,300)",
    fontsize=14,
)
for r, idx in enumerate(sample_indices):
    orig = (Xc[:, idx] + mean_vec[:, 0]).reshape(img_size)
    ax = axes_rec[r, 0]
    ax.imshow(orig, cmap="gray")
    ax.set_title("orig idx={}".format(idx))
    ax.axis("off")
    for c, k in enumerate(ks, start=1):
        A = U[:, :k]
        coeffs_k = coeffs_full[:k, :]
        Xc_recon_k = A.dot(coeffs_k)
        recon = (Xc_recon_k[:, idx] + mean_vec[:, 0]).reshape(img_size)
        ax = axes_rec[r, c]
        ax.imshow(recon, cmap="gray")
        ax.set_title("k={}".format(k))
        ax.axis("off")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_rec.show()

ks_snr = [1, 2, 5, 10, 20, 40, 100, 200, 300, n]
snr_vals = []
for k in ks_snr:
    A = U[:, :k]
    coeffs_k = coeffs_full[:k, :]
    Xc_recon_k = A.dot(coeffs_k)
    X_recon_k = Xc_recon_k + mean_vec
    num = np.linalg.norm(X_recon_k, "fro") ** 2
    den = np.linalg.norm(X_recon_k - X_arr, "fro") ** 2
    snr_k = 10 * np.log10(num / (den + 1e-15))
    snr_vals.append(snr_k)
    print(f"SNR for k={k}: {snr_k:.4f} dB")

fig_snr, ax_snr = plt.subplots(figsize=(7, 4))
ax_snr.plot(ks_snr, snr_vals, marker="o")
ax_snr.set_xlabel("k (number of eigenfaces)")
ax_snr.set_ylabel("SNR (dB)")
ax_snr.set_title("SNR vs k")
ax_snr.grid(True)
fig_snr.tight_layout()
fig_snr.show()

plt.show()
