# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:04:11 2025

@author: Neal LONG

Use GridSearchCV to select 
   1) the best 'C' from C_candidates, and
   2) the best 'gamma' from gamma_candidates 
in combination for SVC model on the training data (X, y)

Note:  
    0. Set random_state=0 for all models and cross-validation procedures.
    1. Evaluate the compare the performance by micro recall 
    2. Use 5-fold Stratified CV 

"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
gamma_candidates = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
C_candidates = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,1000]


#++insert your code below++