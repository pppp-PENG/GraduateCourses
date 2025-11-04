# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:04:38 2025

@author: Neal LONG

Evaluate the learning curve using 10-fold cross-validation for three different models on the dataset X and y:
1. RBF-SVM: SVC model with RBF kernel and gamma=0.001
2. LinearSVC: LinearSVC model with C=0.001 and dual=True
3. DecisionTree: DecisionTreeClassifier model 
Focus on the learning capacity of these models using the following training sizes:
10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, and 100% of all training examples.

Note:
    0. Set random_state=0 for all models and cross-validation procedures.
    1. Use 10-fold stratified cross-validation.
    2. Generate learning curves for the specified training sizes.
    3. Refer to "learning_curve.py" from Week 7 for guidance.
    4. Utilize accuracy as the scoring metric.
    5. The plot title should indicate the model name and your student ID (stu_id) formatted as: f"Learning Curve of {model} by {stu_id}".

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC,LinearSVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

def plot_learning_curve(estimator, title, X, y, givenTrainSizes, scoring = 'accuracy', cv = None):
    
    fig =  plt.figure(1, figsize=(6, 6))
    ax = fig.add_subplot(111)

    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
   
    
    # read the help of learning_curve, and call learning_curve with proper paramters
    train_sizes, train_scores, test_scores = learning_curve(estimator,X,y,
                                                            scoring=scoring,
                                                            cv=cv,
                                                            train_sizes=givenTrainSizes,
                                                            random_state=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.ylim((0.5,1.1))

    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    for xy in zip(train_sizes, test_scores_mean):                                       # <--
        ax.annotate('%s' % round(xy[1],2), xy=xy, textcoords='data')
    plt.legend(loc="best")
    plt.show()



digits = load_digits()
X, y = digits.data, digits.target

# %%
#++insert your code below++
