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

    scores = flow.evaluate(X, y, verbose=True)
    print(scores.mean(), scores.std())

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

        result = flow.evaluate(X, y)

        print result
        print '\n\n=============\n\n'
