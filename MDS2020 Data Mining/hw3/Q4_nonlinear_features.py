# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 00:35:06 2025

@author: Neal

Complete the code to assess the sensitivity to outliers of the following three models:
    1. Plain Logistic Regression model without any penalty, trained on the 2-D features provided.
    2. Plain Logistic Regression model without any penalty, trained on the extended 3-D features.
    3. LinearSVC model with "max_iter"=1000000 and dual=True, trained on the extended 3-D features.

Note:
    0. Start with the provided 2-D binary classification training data (X,y) 
        using 'sepal width (cm)' and 'petal width (cm)' with one added outlier. 
        Extend X to 3-D features by adding the square of 'sepal width (cm)'.
    1. Train the Logistic Regression and LinearSVC models on 
        the extended 3-D training dataset as specified, respectively.
    2. Refer to and modify the existing function plot_mesh_labels() to 
        create a new function plot_mesh_labels_3d() that can plot the 
        decision boundary of the model trained on the extended 3-D features 
        while using 2-D coordinates ('sepal width (cm)' and 'petal width (cm)').
    3. Visualize the results of the two trained models as follows:
        1) Use the modified plot_mesh_labels() function to plot the decision boundary.
        2) Plot the true labels of the training examples.
        3) Highlight the outlier.
        4) Mark any errors.
    4. The title of each plot should indicate the model name and your student ID (stu_id) formatted as: 
        f"Decision boundary of a {model} by {stu_id}".
    5. Set random_state=0 for all models 

Hint: Consider using np.c_ for feature extension.

"""
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

def plot_mesh_labels(plt, model, x_min, x_max, y_min, y_max, x_label, y_label, step = 0.01):
    # plot the  labels of points predicted by model in the mesh area
    # [x_min, x_max]x[y_min, y_max] with plt.
   
    # Generate the points in in the mesh [x_min, x_max]x[y_min, y_max] with step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    # merged into matrix of points, row <-> point
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    # predict the lables
    mesh_labels = model.predict(mesh_points)
    # reshape the labels to 1-d array
    mesh_labels_array = mesh_labels.reshape(xx.shape)

    # Put the result into a color plot    
    plt.pcolormesh(xx, yy, mesh_labels_array, cmap=plt.cm.RdYlBu, shading='auto')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(x_label)
    plt.ylabel(y_label)


    
# Load data
iris = load_iris()

# We only consider binary class classification problem (first 100 examples)
# and 2-d feature, i.e., 'sepal width (cm)' and, 'petal width (cm)'. 
X = iris.data[:101, [1,3]]
Y = iris.target[:101]

# In addition, we also add one outlier to test the sensitivity
# of plain Logistic Regression and LinearSVC model
Y[100] = 1
X[100,:] = [4, 0.7]


# Define the plot area and step size
x_min = 0
x_max = X[:, 0].max() + 0.5
y_min = 0
y_max = X[:, 1].max() + 0.5
h = 0.01  # step size in the mesh

# %% ++modify the code below regarding definition of Logistic Regression as required above++
# Visualize the results of a default Logistic Regression model trained on orginal 2-d features
lr_model = LogisticRegression(random_state=0)
lr_model.fit(X, Y)
Y_pred = lr_model.predict(X)
X_err = X[Y != Y_pred]
Y_err = Y[Y != Y_pred]
print("\nThere are {} errors/mismatches in Logistic Regression".format(sum(Y != Y_pred)))

# plot decision boundary
plt.figure(1, figsize=(9, 9))
plot_mesh_labels(plt, lr_model, x_min ,x_max, y_min, y_max, 
                 iris.feature_names[1], iris.feature_names[3], h)

# plot the true label of training examples
plt.scatter(X[:, 0], X[:, 1], c=Y, facecolor='k', cmap=plt.cm.RdYlBu)
# highlight the outlier
plt.scatter(X[100, 0], X[100, 1],  edgecolor='k')
# mark the errors 
plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', c='k')


# set the title and axis
plt.title("Decision boundary of a Logistic Regression")
plt.axis("tight")
plt.show()

#%% ++insert your code below++