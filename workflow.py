import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score

from logger import logger


__all__ = ['NCV_DEFAULT_TRIALS', 'NestedCrossValidation']


NCV_DEFAULT_TRIALS = 30

NCV_KEY_CONFIG_MODEL = 'model'
NCV_KEY_CONFIG_GRID = 'grid'
NCV_KEY_CONFIG_OUTER = 'outer'
NCV_KEY_CONFIG_INNER = 'inner'
NCV_KEY_CONFIG_TRIALS = 'trials'

NCV_KEY_EVAL_ESTIMATOR = 'estimator',
NCV_KEY_EVAL_FEATURES = 'features',
NCV_KEY_EVAL_LABELS = 'labels'
NCV_KEY_EVAL_NESTED = 'nested'
NCV_KEY_EVAL_VERBOSE = 'verbose',

NCV_ERRORS_PARAMS = {
    NCV_KEY_CONFIG_MODEL: 'The estimator model is not set.',
    NCV_KEY_CONFIG_GRID: 'The search parameters grid is not set.',
    NCV_KEY_CONFIG_INNER: 'The inner loop cross-validator is not set.',
    NCV_KEY_CONFIG_OUTER: 'The outer loop cross-validator is not set.'
}


class NestedCrossValidation(object):
    def _parse(self, params):
        try:
            o = params[0]
        except IndexError:
            raise AttributeError('Error: you need to provide a class for your estimator or cross-validator.')

        args = []
        kwargs = {}

        for i in (1, 2):
            try:
                arg_ = params[i]
                type_ = type(arg_)

                if type_ == dict:
                    kwargs = arg_
                if type_ == list or type_ == tuple:
                    args = arg_

            except IndexError:
                pass

        return (o, args, kwargs)

    def _assess(self):
        for key, msg in NCV_ERRORS_PARAMS.items():
            if self._params[key] is None:
                raise AttributeError(msg)

    def _reset(self):
        self._eparams = {
            NCV_KEY_EVAL_VERBOSE: False,
            NCV_KEY_EVAL_ESTIMATOR: None,
            NCV_KEY_EVAL_FEATURES: None,
            NCV_KEY_EVAL_LABELS: None
        }

    def _trial(self, i):
        verbose = self._eparams[NCV_KEY_EVAL_VERBOSE]

        estimator = self._eparams[NCV_KEY_EVAL_ESTIMATOR]
        grid = self._params[NCV_KEY_CONFIG_GRID]

        cv, args, kwargs = self._params[NCV_KEY_CONFIG_INNER]
        kwargs['random_state'] = i
        inner = cv(*args, **kwargs)

        cv, args, kwargs = self._params[NCV_KEY_CONFIG_OUTER]
        kwargs['random_state'] = i
        outer = cv(*args, **kwargs)

        X = self._eparams[NCV_KEY_EVAL_FEATURES]
        y = self._eparams[NCV_KEY_EVAL_LABELS]

        clf = GridSearchCV(estimator=estimator, param_grid=grid, cv=inner, iid=True)
        clf.fit(X, y)

        if self._eparams[NCV_KEY_EVAL_NESTED] is True:
            scores = cross_val_score(clf, X=X, y=y, cv=outer)
            score = scores.mean()
        else:
            score = clf.best_score_

        if verbose:
            logger.info((i, score))

        return score

    def __init__(self):
        self._params = {
            NCV_KEY_CONFIG_MODEL: None,
            NCV_KEY_CONFIG_GRID: None,
            NCV_KEY_CONFIG_INNER: None,
            NCV_KEY_CONFIG_OUTER: None,
            NCV_KEY_CONFIG_TRIALS: NCV_DEFAULT_TRIALS
        }

    def evaluate(self, X, y, nested=True, verbose=False):
        self._assess()
        self._reset()

        self._eparams[NCV_KEY_EVAL_FEATURES] = X
        self._eparams[NCV_KEY_EVAL_LABELS] = y

        self._eparams[NCV_KEY_EVAL_NESTED] = nested

        self._eparams[NCV_KEY_EVAL_VERBOSE] = verbose

        model, args, kwargs = self._params[NCV_KEY_CONFIG_MODEL]
        self._eparams[NCV_KEY_EVAL_ESTIMATOR] = model(*args, **kwargs)

        trials = self._params[NCV_KEY_CONFIG_TRIALS]
        scores = np.zeros((trials, 1), dtype=np.float64)

        for i in range(trials):
            scores[i, :] = self._trial(i)

        self._reset()

        return scores

    @property
    def model(self):
        return self._params[NCV_KEY_CONFIG_MODEL]

    @model.setter
    def model(self, params):
        model, args, kwargs = self._parse(params)
        self._params[NCV_KEY_CONFIG_MODEL] = (model, args, kwargs)

    @property
    def grid(self):
        return self._params[NCV_KEY_CONFIG_GRID]

    @grid.setter
    def grid(self, params):
        self._params[NCV_KEY_CONFIG_GRID] = params

    @property
    def inner(self):
        return self._params[NCV_KEY_CONFIG_INNER]

    @inner.setter
    def inner(self, params):
        cv, args, kwargs = self._parse(params)
        self._params[NCV_KEY_CONFIG_INNER] = (cv, args, kwargs)

    @property
    def outer(self):
        return self._params[NCV_KEY_CONFIG_OUTER]

    @outer.setter
    def outer(self, params):
        cv, args, kwargs = self._parse(params)
        self._params[NCV_KEY_CONFIG_OUTER] = (cv, args, kwargs)

    @property
    def trials(self):
        return self._params[NCV_KEY_CONFIG_TRIALS]

    @trials.setter
    def trials(self, n):
        self._params[NCV_KEY_CONFIG_TRIALS] = n
