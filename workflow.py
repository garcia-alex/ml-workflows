import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler  # , PolynomialFeatures
from sklearn.model_selection import GridSearchCV, cross_val_score

from logger import logger


__all__ = ['NCV_DEFAULT_TRIALS', 'NestedCrossValidation']


NCV_DEFAULT_TRIALS = 10

NCV_KEY_CONFIG_DIMR = 'dimr'
NCV_KEY_CONFIG_MODEL = 'model'
NCV_KEY_CONFIG_HYPER = 'hyper'
NCV_KEY_CONFIG_OUTER = 'outer'
NCV_KEY_CONFIG_INNER = 'inner'
NCV_KEY_CONFIG_TRIALS = 'trials'

NCV_KEY_EVAL_DIMR = 'dimr'
NCV_KEY_EVAL_ESTIMATOR = 'estimator'
NCV_KEY_EVAL_FEATURES = 'features'
NCV_KEY_EVAL_LABELS = 'labels'
NCV_KEY_EVAL_NESTED = 'nested'
NCV_KEY_EVAL_SCALE = 'scale'
NCV_KEY_EVAL_IID = 'iid'
NCV_KEY_EVAL_VERBOSE = 'verbose'

NCV_PARAMS_WARNINGS = {
    NCV_KEY_CONFIG_INNER: 'The inner loop cv splitter is not set, will use sklearn default.',
    NCV_KEY_CONFIG_OUTER: 'The outer loop cv splitter is not set, will use sklearn default.',
    NCV_KEY_CONFIG_DIMR: 'The dimensionality reducer is not set, will skip operation.'
}

NCV_PARAMS_ERRORS = {
    NCV_KEY_CONFIG_MODEL: 'The estimator model is not set.',
    NCV_KEY_CONFIG_HYPER: 'The hyper parameters grid is not set.'
}

NCV_PIPELINE_KEY_IMPUTER = 'imputer'
NCV_PIPELINE_KEY_SCALER = 'scaler'
NCV_PIPELINE_KEY_POLY = 'poly'
NCV_PIPELINE_KEY_DIMR = 'dimr'
NCV_PIPELINE_KEY_MODEL = 'model'

NCV_IMPUTER_STRATEGY_MEDIAN = 'median'


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
        for key, msg in NCV_PARAMS_ERRORS.items():
            if self._params[key] is None:
                raise AttributeError(msg)

        verbose = self._eparams[NCV_KEY_EVAL_VERBOSE]

        if verbose:
            for key, msg in NCV_PARAMS_WARNINGS.items():
                if self._params[key] is None:
                    logger.warn(msg)

    def _reset(self):
        self._eparams = {
            NCV_KEY_EVAL_FEATURES: None,
            NCV_KEY_EVAL_LABELS: None,

            NCV_KEY_EVAL_NESTED: True,
            NCV_KEY_EVAL_SCALE: True,
            NCV_KEY_EVAL_IID: True,

            NCV_KEY_EVAL_VERBOSE: False,

            NCV_KEY_EVAL_DIMR: None,
            NCV_KEY_EVAL_ESTIMATOR: None
        }

    def _pipeline(self):
        dimr = self._eparams[NCV_KEY_EVAL_DIMR]
        scale = self._eparams[NCV_KEY_EVAL_SCALE]
        estimator = self._eparams[NCV_KEY_EVAL_ESTIMATOR]

        params = [
            (NCV_PIPELINE_KEY_IMPUTER, SimpleImputer(strategy=NCV_IMPUTER_STRATEGY_MEDIAN))
        ]

        if scale:
            params.append((NCV_PIPELINE_KEY_SCALER, StandardScaler()))

        if dimr:
            params.append((NCV_PIPELINE_KEY_DIMR, None))

        params.append((NCV_PIPELINE_KEY_MODEL, estimator))

        pipeline = Pipeline(params)

        return pipeline

    def _grid(self):
        params = {}

        dimr = self._eparams[NCV_KEY_EVAL_DIMR]

        if dimr:
            params[NCV_KEY_CONFIG_DIMR] = [dimr]

        hyper = self._params[NCV_KEY_CONFIG_HYPER] or {}

        for key in (NCV_KEY_CONFIG_MODEL, NCV_KEY_CONFIG_DIMR):
            for param, values in hyper.get(key, {}).items():
                name = f'{key}__{param}'
                params[name] = values

        grid = [params]

        return grid

    def _splitter(self, key):
        if self._params[key] is not None:
            cv, args, kwargs = self._params[key]
            splitter = cv(*args, **kwargs)
        else:
            splitter = 5

        return splitter

    def _trial(self, pipeline, grid):
        X = self._eparams[NCV_KEY_EVAL_FEATURES]
        y = self._eparams[NCV_KEY_EVAL_LABELS]

        inner = self._splitter(NCV_KEY_CONFIG_INNER)
        iid = self._eparams[NCV_KEY_EVAL_IID]

        clf = GridSearchCV(pipeline, param_grid=grid, cv=inner, iid=iid)
        clf.fit(X, y)

        if self._eparams[NCV_KEY_EVAL_NESTED] is True:
            outer = self._splitter(NCV_KEY_CONFIG_OUTER)
            scores = cross_val_score(clf, X=X, y=y, cv=outer)
            score = scores.mean()
        else:
            score = clf.best_score_

        return score

    def __init__(self):
        self._params = {
            NCV_KEY_CONFIG_DIMR: None,
            NCV_KEY_CONFIG_MODEL: None,
            NCV_KEY_CONFIG_HYPER: None,
            NCV_KEY_CONFIG_INNER: None,
            NCV_KEY_CONFIG_OUTER: None,
            NCV_KEY_CONFIG_TRIALS: NCV_DEFAULT_TRIALS
        }

    def evaluate(self, X, y, scale=True, nested=True, iid=True, verbose=False):
        self._reset()

        self._eparams[NCV_KEY_EVAL_FEATURES] = X
        self._eparams[NCV_KEY_EVAL_LABELS] = y

        self._eparams[NCV_KEY_EVAL_SCALE] = scale
        self._eparams[NCV_KEY_EVAL_NESTED] = nested
        self._eparams[NCV_KEY_EVAL_IID] = iid

        self._eparams[NCV_KEY_EVAL_VERBOSE] = verbose

        self._assess()

        try:
            dimr, args, kwargs = self._params[NCV_KEY_CONFIG_DIMR]
            self._eparams[NCV_KEY_EVAL_DIMR] = dimr(*args, **kwargs)
        except TypeError:
            pass

        model, args, kwargs = self._params[NCV_KEY_CONFIG_MODEL]
        self._eparams[NCV_KEY_EVAL_ESTIMATOR] = model(*args, **kwargs)

        trials = self._params[NCV_KEY_CONFIG_TRIALS]
        scores = np.zeros((trials, 1), dtype=np.float64)

        pipeline = self._pipeline()
        grid = self._grid()

        for i in range(trials):
            scores[i, :] = self._trial(pipeline, grid)

        self._reset()

        return scores

    @property
    def dimr(self):
        return self._params[NCV_KEY_CONFIG_MODEL]

    @dimr.setter
    def dimr(self, params):
        dimr, args, kwargs = self._parse(params)
        self._params[NCV_KEY_CONFIG_DIMR] = (dimr, args, kwargs)

    @property
    def model(self):
        return self._params[NCV_KEY_CONFIG_MODEL]

    @model.setter
    def model(self, params):
        model, args, kwargs = self._parse(params)
        self._params[NCV_KEY_CONFIG_MODEL] = (model, args, kwargs)

    @property
    def hyper(self):
        return self._params[NCV_KEY_CONFIG_HYPER]

    @hyper.setter
    def hyper(self, params):
        self._params[NCV_KEY_CONFIG_HYPER] = params

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
