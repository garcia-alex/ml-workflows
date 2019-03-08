from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit
)

from workflow import NestedCrossValidation


if __name__ == '__main__':
    ncv = NestedCrossValidation()
    ncv.model = (SVC, dict(kernel='rbf'))
    ncv.grid = {
        'C': [1, 10, 100],
        'gamma': [.01, .1]
    }

    kfold = (KFold, dict(n_splits=4, shuffle=True))
    rkfold = (RepeatedKFold, dict(n_splits=2, n_repeats=2))
    skfold = (StratifiedKFold, dict(n_splits=3))
    shuffle = (ShuffleSplit, dict(n_splits=5, test_size=0.25))
    sshuffle = (StratifiedShuffleSplit, dict(n_splits=5, test_size=0.25))

    iterators = (kfold, rkfold, skfold, shuffle, sshuffle)

    for iterator in iterators:
        print(iterator)

        ncv.outer = iterator
        ncv.inner = iterator

        iris = load_iris()
        X = iris.data
        y = iris.target

        scores = ncv.evaluate(X, y)

        print(scores.mean(), scores.std())
