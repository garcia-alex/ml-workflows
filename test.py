import json

import numpy as np
from scipy.stats import expon

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit
)
from sklearn.svm import SVC

from ml_workflow.workflow import EvaluationWorkflow
from ml_workflow.pipelines import Pipeline


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, Pipeline):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    flow = EvaluationWorkflow()

    flow.model = (SVC, dict(kernel='rbf'))
    flow.trials = 3

    flow.hyper = {
        'model': {
            '__randomized__': True,
            'C': expon(scale=100), # [1, 10, 100],
            'gamma':expon(scale=.1) # [.01, .1]
        }
    }

    results = flow.evaluate(X, y, verbose=True)
    print(results)

    print('\n\n=============\n\n')

    pca = (PCA, dict(iterated_power=7))
    ipca = (IncrementalPCA, dict())
    kpca = (KernelPCA, dict(kernel='rbf'))

    flow.dimrs = [pca, ipca, kpca]

    kfold = (KFold, dict(n_splits=4, shuffle=True))
    rkfold = (RepeatedKFold, dict(n_splits=2, n_repeats=2))
    skfold = (StratifiedKFold, dict(n_splits=3))
    shuffle = (ShuffleSplit, dict(n_splits=5, test_size=0.25))
    sshuffle = (StratifiedShuffleSplit, dict(n_splits=5, test_size=0.25))

    splitters = (kfold, rkfold, skfold, shuffle, sshuffle)

    for splitter in splitters:
        print(splitter)

        flow.outer = splitter
        flow.inner = splitter

        results = flow.evaluate(X, y)

        print(results)
        print('\n\n=============\n\n')
