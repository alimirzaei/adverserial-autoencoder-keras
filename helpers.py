# -*- coding: utf-8 -*-
from sklearn.neighbors.kde import KernelDensity
import numpy as np
from sklearn.model_selection import GridSearchCV


def approximateLogLiklihood(x_generated, x_test):
    x_generated = np.array(x_generated).reshape((len(x_generated),-1))
    x_test = np.array(x_test).reshape((len(x_test),-1))
    # use grid search cross-validation to optimize the bandwidth
    print "new"
    params = {'bandwidth': np.logspace(-4, -1, 5)}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=4)
    grid.fit(x_generated)
    print grid.best_params_
    kde = grid.best_estimator_
    scores = kde.score_samples(x_test)
    return np.sum(scores)/len(scores)