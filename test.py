from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import csv
import os.path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print(X_train)
print(y_train)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


x_train_p = df.iloc[0:start, 2:-1]
x_train = x_train_p.to_numpy()
y_train_p = df.iloc[0:start, [-1]]
y_train = y_train_p.to_numpy().ravel()

x_test_p = df.iloc[start:n_obs, 2:-1]
x_test = x_test_p.to_numpy()
y_test_p = df.iloc[start:n_obs, [-1]]
y_test = y_test_p.to_numpy().ravel()

def try_boosted_dt():
    real_test_errors = []
    discrete_test_errors = []

    for real_test_predict, discrete_train_predict in zip(
            bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
        real_test_errors.append(
            1. - accuracy_score(real_test_predict, y_test))
        discrete_test_errors.append(
            1. - accuracy_score(discrete_train_predict, y_test))

    n_trees_discrete = len(bdt_discrete)
    n_trees_real = len(bdt_real)

    # Boosting might terminate early, but the following arrays are always
    # n_estimators long. We crop them to the actual number of trees here:
    discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
    real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
    discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(range(1, n_trees_discrete + 1),
            discrete_test_errors, c='black', label='SAMME')
    plt.plot(range(1, n_trees_real + 1),
            real_test_errors, c='black',
            linestyle='dashed', label='SAMME.R')
    plt.legend()
    plt.ylim(0.18, 0.62)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees')

    plt.subplot(132)
    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
            "b", label='SAMME', alpha=.5)
    plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
            "r", label='SAMME.R', alpha=.5)
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Number of Trees')
    plt.ylim((.2,
            max(real_estimator_errors.max(),
                discrete_estimator_errors.max()) * 1.2))
    plt.xlim((-20, len(bdt_discrete) + 20))

    plt.subplot(133)
    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
            "b", label='SAMME')
    plt.legend()
    plt.ylabel('Weight')
    plt.xlabel('Number of Trees')
    plt.ylim((0, discrete_estimator_weights.max() * 1.2))
    plt.xlim((-20, n_trees_discrete + 20))

    # prevent overlapping y-axis labels
    plt.subplots_adjust(wspace=0.25)
    plt.show()