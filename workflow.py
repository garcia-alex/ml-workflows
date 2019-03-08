import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler  # , PolynomialFeatures
from sklearn.model_selection import GridSearchCV, cross_val_score

from logger import logger


__all__ = ['EW_DEFAULT_TRIALS', 'EvaluationWorkflow']


EW_DEFAULT_TRIALS = 10

EW_KEY_CONFIG_DIMR = 'dimr'
EW_KEY_CONFIG_MODEL = 'model'
EW_KEY_CONFIG_HYPER = 'hyper'
EW_KEY_CONFIG_OUTER = 'outer'
EW_KEY_CONFIG_INNER = 'inner'
EW_KEY_CONFIG_TRIALS = 'trials'

EW_KEY_EVAL_DIMR = 'dimr'
EW_KEY_EVAL_ESTIMATOR = 'estimator'
EW_KEY_EVAL_FEATURES = 'features'
EW_KEY_EVAL_LABELS = 'labels'
EW_KEY_EVAL_NESTED = 'nested'
EW_KEY_EVAL_SCALE = 'scale'
EW_KEY_EVAL_IID = 'iid'
EW_KEY_EVAL_VERBOSE = 'verbose'

EW_PARAMS_WARNINGS = {
    EW_KEY_CONFIG_INNER: 'The inner loop cv splitter is not set, will use sklearn default.',
    EW_KEY_CONFIG_OUTER: 'The outer loop cv splitter is not set, will use sklearn default.',
    EW_KEY_CONFIG_DIMR: 'The dimensionality reducer is not set, will skip operation.'
}

EW_PARAMS_ERRORS = {
    EW_KEY_CONFIG_MODEL: 'The estimator model is not set.',
    EW_KEY_CONFIG_HYPER: 'The hyper parameters grid is not set.'
}

EW_PIPELINE_KEY_IMPUTER = 'imputer'
EW_PIPELINE_KEY_SCALER = 'scaler'
EW_PIPELINE_KEY_POLY = 'poly'
EW_PIPELINE_KEY_DIMR = 'dimr'
EW_PIPELINE_KEY_MODEL = 'model'

EW_IMPUTER_STRATEGY_MEDIAN = 'median'


class EvaluationWorkflow(object):
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
        for key, msg in EW_PARAMS_ERRORS.items():
            if self._params[key] is None:
                raise AttributeError(msg)

        verbose = self._eparams[EW_KEY_EVAL_VERBOSE]

        if verbose:
            for key, msg in EW_PARAMS_WARNINGS.items():
                if self._params[key] is None:
                    logger.warn(msg)

    def _reset(self):
        self._eparams = {
            EW_KEY_EVAL_FEATURES: None,
            EW_KEY_EVAL_LABELS: None,

            EW_KEY_EVAL_NESTED: True,
            EW_KEY_EVAL_SCALE: True,
            EW_KEY_EVAL_IID: True,

            EW_KEY_EVAL_VERBOSE: False,

            EW_KEY_EVAL_DIMR: None,
            EW_KEY_EVAL_ESTIMATOR: None
        }

    def _pipeline(self):
        dimr = self._eparams[EW_KEY_EVAL_DIMR]
        scale = self._eparams[EW_KEY_EVAL_SCALE]
        estimator = self._eparams[EW_KEY_EVAL_ESTIMATOR]

        params = [
            (EW_PIPELINE_KEY_IMPUTER, SimpleImputer(strategy=EW_IMPUTER_STRATEGY_MEDIAN))
        ]

        if scale:
            params.append((EW_PIPELINE_KEY_SCALER, StandardScaler()))

        if dimr:
            params.append((EW_PIPELINE_KEY_DIMR, None))

        params.append((EW_PIPELINE_KEY_MODEL, estimator))

        pipeline = Pipeline(params)

        return pipeline

    def _grid(self):
        params = {}

        dimr = self._eparams[EW_KEY_EVAL_DIMR]

        if dimr:
            params[EW_KEY_CONFIG_DIMR] = [dimr]

        hyper = self._params[EW_KEY_CONFIG_HYPER] or {}

        for key in (EW_KEY_CONFIG_MODEL, EW_KEY_CONFIG_DIMR):
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
        X = self._eparams[EW_KEY_EVAL_FEATURES]
        y = self._eparams[EW_KEY_EVAL_LABELS]

        inner = self._splitter(EW_KEY_CONFIG_INNER)
        iid = self._eparams[EW_KEY_EVAL_IID]

        clf = GridSearchCV(pipeline, param_grid=grid, cv=inner, iid=iid)
        clf.fit(X, y)

        if self._eparams[EW_KEY_EVAL_NESTED] is True:
            outer = self._splitter(EW_KEY_CONFIG_OUTER)
            scores = cross_val_score(clf, X=X, y=y, cv=outer)
            score = scores.mean()
        else:
            score = clf.best_score_

        return score

    def __init__(self):
        self._params = {
            EW_KEY_CONFIG_DIMR: None,
            EW_KEY_CONFIG_MODEL: None,
            EW_KEY_CONFIG_HYPER: None,
            EW_KEY_CONFIG_INNER: None,
            EW_KEY_CONFIG_OUTER: None,
            EW_KEY_CONFIG_TRIALS: EW_DEFAULT_TRIALS
        }

    def evaluate(self, X, y, scale=True, nested=True, iid=True, verbose=False):
        self._reset()

        self._eparams[EW_KEY_EVAL_FEATURES] = X
        self._eparams[EW_KEY_EVAL_LABELS] = y

        self._eparams[EW_KEY_EVAL_SCALE] = scale
        self._eparams[EW_KEY_EVAL_NESTED] = nested
        self._eparams[EW_KEY_EVAL_IID] = iid

        self._eparams[EW_KEY_EVAL_VERBOSE] = verbose

        self._assess()

        try:
            dimr, args, kwargs = self._params[EW_KEY_CONFIG_DIMR]
            self._eparams[EW_KEY_EVAL_DIMR] = dimr(*args, **kwargs)
        except TypeError:
            pass

        model, args, kwargs = self._params[EW_KEY_CONFIG_MODEL]
        self._eparams[EW_KEY_EVAL_ESTIMATOR] = model(*args, **kwargs)

        trials = self._params[EW_KEY_CONFIG_TRIALS]
        scores = np.zeros((trials, 1), dtype=np.float64)

        pipeline = self._pipeline()
        grid = self._grid()

        for i in range(trials):
            scores[i, :] = self._trial(pipeline, grid)

        self._reset()

        return scores

    @property
    def dimr(self):
        return self._params[EW_KEY_CONFIG_MODEL]

    @dimr.setter
    def dimr(self, params):
        dimr, args, kwargs = self._parse(params)
        self._params[EW_KEY_CONFIG_DIMR] = (dimr, args, kwargs)

    @property
    def model(self):
        return self._params[EW_KEY_CONFIG_MODEL]

    @model.setter
    def model(self, params):
        model, args, kwargs = self._parse(params)
        self._params[EW_KEY_CONFIG_MODEL] = (model, args, kwargs)

    @property
    def hyper(self):
        return self._params[EW_KEY_CONFIG_HYPER]

    @hyper.setter
    def hyper(self, params):
        self._params[EW_KEY_CONFIG_HYPER] = params

    @property
    def inner(self):
        return self._params[EW_KEY_CONFIG_INNER]

    @inner.setter
    def inner(self, params):
        cv, args, kwargs = self._parse(params)
        self._params[EW_KEY_CONFIG_INNER] = (cv, args, kwargs)

    @property
    def outer(self):
        return self._params[EW_KEY_CONFIG_OUTER]

    @outer.setter
    def outer(self, params):
        cv, args, kwargs = self._parse(params)
        self._params[EW_KEY_CONFIG_OUTER] = (cv, args, kwargs)

    @property
    def trials(self):
        return self._params[EW_KEY_CONFIG_TRIALS]

    @trials.setter
    def trials(self, n):
        self._params[EW_KEY_CONFIG_TRIALS] = n
