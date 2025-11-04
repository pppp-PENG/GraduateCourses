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
    V = (255. * V) / (np.asmatrix(np.ones((V.shape[0], 1))) * max_val) / 100.
    return V

def read_orl(img_size=(112, 92)):
    """
    Read face image data from the ORL database. The matrix's shape is 2576 (pixels) x 400 (faces). 
    
    Step through each subject and each image. Reduce the size of the images by a factor of 0.5. 
    
    Return the ORL faces data matrix. 
    """
    print("Reading ORL faces database")
    dir = join('ORL_faces', 's')
    V = np.matrix(np.zeros((img_size[0] * img_size[1], 400)))
    for subject in range(40):
        for image in range(10):
            im = open(join(dir + str(subject + 1), str(image + 1) + ".pgm"))
            # reduce the size of the image
            im = im.resize(img_size[::-1])
            V[:, 10 * subject + image] = np.asmatrix(np.asarray(im).flatten()).T
    return V

def cal_snr(img_origin, img_recon):
    ratio = np.sum(np.power(img_recon, 2)) / np.sum(np.power((img_origin-img_recon), 2))
    return 10 * np.log10(ratio)

img_size = (112, 92)
X = read_orl(img_size)
X_processed = preprocess(X)

# plot the eigenfaces, you may use plt.imshow(X[:, i].reshape(*img_size), cmap='gray') to visualize the data

# plot original img & reconstructed img

# calculate snr