from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.decomposition import PCA, NMF
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

    ncv.hyper = {
        'dimr': {
            'n_components': [1, 2, 3]
        },
        'model': {
            'C': [1, 10, 100],
            'gamma': [.01, .1]
        }
    }

    pca = (PCA, dict(iterated_power=7))
    nmf = (NMF, dict())

    kfold = (KFold, dict(n_splits=4, shuffle=True))
    rkfold = (RepeatedKFold, dict(n_splits=2, n_repeats=2))
    skfold = (StratifiedKFold, dict(n_splits=3))
    shuffle = (ShuffleSplit, dict(n_splits=5, test_size=0.25))
    sshuffle = (StratifiedShuffleSplit, dict(n_splits=5, test_size=0.25))

    dimrs = (pca, nmf)
    splitters = (kfold, rkfold, skfold, shuffle, sshuffle)

    for dimr in dimrs:
        ncv.dimr = dimr

        for splitter in splitters:
            print(dimr, splitter)

            ncv.outer = splitter
            ncv.inner = splitter

            iris = load_iris()
            X = iris.data
            y = iris.target

            scores = ncv.evaluate(X, y)

            print(scores.mean(), scores.std())
