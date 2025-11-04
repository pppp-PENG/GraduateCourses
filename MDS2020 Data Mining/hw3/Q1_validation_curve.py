# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:30:33 2025

@author: Neal LONG
Ref:
    1.plt.semilogx: https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.semilogx.html

Please perform 10-fold cross-validation to identify the optimal gamma for the SVC model from the provided gamma_candidates.

Note:
    0. Set random_state=0 for all models and cross-validation procedures.
    1. Use plt.semilogx to visualize the training and cross-validation score curve, as the range of gamma in "gamma_candidates" spans different scales.
    2. The plot title should reflect the number of folds (k) used in cross-validation and your student ID (stu_id) formatted as: f"Validation Curve based on {k}-fold CV by {stu_id}".
    3. Utilize accuracy as the scoring metric.
    4. Implement 10-fold stratified cross-validation.
    5. Refer to "fitting_graph.ipynb" from Week 7 for guidance.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedKFold

X, y = load_digits(return_X_y=True)
gamma_candidates = np.logspace(-6, -1, 5)
plt.plot(gamma_candidates, [1,2,3,2,1], marker='o')
plt.show()
plt.title("Test of Plot Title")
plt.semilogx(gamma_candidates, [1,2,3,2,1], marker='o')
plt.show()

# %% ++insert your code below++