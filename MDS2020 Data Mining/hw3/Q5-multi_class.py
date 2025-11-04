# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:06:05 2025
@author: Neal

Enhance the LogisticRegression model by wrapping it with either OneVsOneClassifier or OneVsRestClassifier to address the 3-class classification task of the IRIS dataset. Specifically, you will:
    1. Select the LogisticRegression model with the optimal solver from the solver_candidates when wrapped by OneVsOneClassifier, and evaluate its average performance using simplified nested cross-validation.
    2. Select the LogisticRegression model with the optimal solver from the solver_candidates when wrapped by OneVsRestClassifier, and evaluate its average performance using simplified nested cross-validation.
    3. Record the performance evaluation results and respond to related questions.
Note:
    0. Set random_state=0 and max_iter=10000 for the LogisticRegression model.
    1. Use all records and attributes from the imported IRIS dataset.
    2. Utilize GridSearchCV to find the best solver for LogisticRegression wrapped by either OneVsOneClassifier or OneVsRestClassifier.
    3. Apply the defined inner_cv and outer_cv but only with one trial as outlined in “simplified_nested_CV.py” from Week 7.
    4. Evaluate and compare model performance based on the average macro-F1 score.

"""


from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Load data
iris = load_iris()
X = iris.data
y = iris.target
solver_candidates = ["lbfgs", "liblinear", "newton-cg"]
inner_cv = StratifiedKFold(5, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(5, shuffle=True, random_state=13)

# %% ++insert your code below++

"""
The first challenge is  how to define the paramter grids for GridSearchCV with 
a wrapped model, because the following code does not work

param_grid = {'solver': solver_candidates}
basic_clf_ovo  = OneVsOneClassifier(LogisticRegression(random_state=0, max_iter=10000))
best_clf_ovo = GridSearchCV(basic_clf_ovo, param_grid, cv=inner_cv, scoring="f1_macro")
best_clf_ovo.fit(X, y)

"""
param_grid = {"estimator__solver": solver_candidates}

basic_clf_ovo = OneVsOneClassifier(LogisticRegression(random_state=0, max_iter=10000))
grid_ovo = GridSearchCV(basic_clf_ovo, param_grid, cv=inner_cv, scoring="f1_macro")
grid_ovo.fit(X, y)
best_clf_ovo = OneVsOneClassifier(
    LogisticRegression(
        solver=grid_ovo.best_params_['estimator__solver'], 
        random_state=0, 
        max_iter=10000
    )
)
scores_ovo = cross_val_score(best_clf_ovo, X, y, cv=outer_cv, scoring="f1_macro")
print("OneVsOne average macro-F1: %.4f" % np.mean(scores_ovo))
print(
    "OneVsOne best solver (full-data GridSearch):",
    grid_ovo.best_params_["estimator__solver"],
)

basic_clf_ovr = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=10000))
grid_ovr = GridSearchCV(basic_clf_ovr, param_grid, cv=inner_cv, scoring="f1_macro")
grid_ovr.fit(X, y)
best_clf_ovr = OneVsRestClassifier(
    LogisticRegression(
        solver=grid_ovr.best_params_['estimator__solver'], 
        random_state=0, 
        max_iter=10000
    )
)
scores_ovr = cross_val_score(best_clf_ovr, X, y, cv=outer_cv, scoring="f1_macro")
print("OneVsRest average macro-F1: %.4f" % np.mean(scores_ovr))
print(
    "OneVsRest best solver (full-data GridSearch):",
    grid_ovr.best_params_["estimator__solver"],
)
