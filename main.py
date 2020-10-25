"""
Perceptron and linear classifier
"""

import numpy as np
from sklearn.linear_model import Perceptron, LogisticRegression

from vis import *
from util import *
from data_loader import DataLoader

DATASET = 'lego_figures'  # ['chinese_mnist', 'lego_figures']
CLASSIFIER = 'linear_classifier'  # ['linear_classifier', 'perceptron']

# Loading data
dl = DataLoader(DATASET, verbose=1, test_split=0.33)
X_train, X_test, y_train, y_test = dl.load()

# Fitting
print("------------")
print("Fitting...")

clf = None
if CLASSIFIER == 'linear_classifier':
    clf = LogisticRegression(verbose=1, n_jobs=-1,
                             random_state=42)
elif CLASSIFIER == 'perceptron':
    clf = Perceptron(verbose=1, n_jobs=-1,
                     penalty='l1', tol=0.1,
                     random_state=42)

clf.fit(X_train, y_train)

print("Done!")

# Validation
print("------------")
print("Validating...")

print()
print("Train scores" )
pred_train = clf.predict(X_train)
print_validation_report(y_train, pred_train)

print()
print("Test scores")
pred_test = clf.predict(X_test)
print_validation_report(y_test, pred_test)

print("Done!")

# Visualization
print("------------")
print("Visualizing...")

visualize(X_train, y_train, pred_train, idx=1, n_samples=1000)
visualize(X_test, y_test, pred_test, idx=2, n_samples=1000)

print("Done!")

show()

