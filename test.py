from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from workflow import NestedCrossValidation


if __name__ == '__main__':
    ncv = NestedCrossValidation()
    ncv.model = (SVC, dict(kernel='rbf'))
    ncv.grid = {
        'C': [1, 10, 100],
        'gamma': [.01, .1]
    }

    ncv.outer = (KFold, dict(n_splits=4, shuffle=True))
    ncv.inner = (KFold, dict(n_splits=4, shuffle=True))

    print(ncv.model)
    print(ncv.outer)
    print(ncv.inner)

    iris = load_iris()
    X = iris.data
    y = iris.target

    scores = ncv.evaluate(X, y, verbose=True)
    print(scores)
