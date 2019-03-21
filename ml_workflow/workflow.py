import numpy as np

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler  # , PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from logger import logger


__all__ = ['EW_DEFAULT_TRIALS', 'EvaluationWorkflow']


EW_DEFAULT_TRIALS = 10

EW_KEY_CONFIG_DIMRS = 'dimrs'
EW_KEY_CONFIG_MODEL = 'model'
EW_KEY_CONFIG_HYPER = 'hyper'
EW_KEY_CONFIG_OUTER = 'outer'
EW_KEY_CONFIG_INNER = 'inner'
EW_KEY_CONFIG_RGS = 'rgs'
EW_KEY_CONFIG_TRIALS = 'trials'

EW_KEY_EVAL_DIMRS = 'dimrs'
EW_KEY_EVAL_MODEL = 'model'
EW_KEY_EVAL_FEATURES = 'features'
EW_KEY_EVAL_LABELS = 'labels'
EW_KEY_EVAL_NESTED = 'nested'
EW_KEY_EVAL_SCALE = 'scale'
EW_KEY_EVAL_IID = 'iid'
EW_KEY_EVAL_VERBOSE = 'verbose'

EW_PARAMS_WARNINGS = {
    EW_KEY_CONFIG_INNER: 'The inner loop cv splitter is not set, will use sklearn default.',
    EW_KEY_CONFIG_OUTER: 'The outer loop cv splitter is not set, will use sklearn default.',
    EW_KEY_CONFIG_DIMRS: 'The dimensionality reducers are not set, will skip feature selection.'
}

EW_PARAMS_ERRORS = {
    EW_KEY_CONFIG_MODEL: 'The model (regression, classification, or clustering) is not set.',
    EW_KEY_CONFIG_HYPER: 'The hyper parameters grid is not set.'
}

EW_PIPELINE_KEY_IMPUTER = 'imputer'
EW_PIPELINE_KEY_SCALER_STANDARD = 'standard'
EW_PIPELINE_KEY_SCALER_ROBUST = 'robust'
EW_PIPELINE_KEY_SCALER_MINMAX = 'minmax'
EW_PIPELINE_KEY_POLY = 'poly'
EW_PIPELINE_KEY_DIMRS = 'dimrs'
EW_PIPELINE_KEY_FEATURES = 'features'
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

            EW_KEY_EVAL_DIMRS: None,
            EW_KEY_EVAL_MODEL: None
        }

    def _dimrs(self):
        if self._params[EW_KEY_CONFIG_DIMRS] is None:
            return None

        dimrs = []

        for dimr in self._params[EW_KEY_CONFIG_DIMRS]:
            try:
                estimator, args, kwargs = self._parse(dimr)
                dimrs.append(estimator(*args, **kwargs))
            except TypeError:
                pass

        dimrs = [(x.__class__.__name__, x) for x in dimrs]

        return dimrs

    def _pipeline(self):
        dimrs = self._eparams[EW_KEY_EVAL_DIMRS]
        scale = self._eparams[EW_KEY_EVAL_SCALE]
        model = self._eparams[EW_KEY_EVAL_MODEL]

        params = [
            (EW_PIPELINE_KEY_IMPUTER, SimpleImputer(strategy=EW_IMPUTER_STRATEGY_MEDIAN))
        ]

        if scale is True:
            params.extend([
                (EW_PIPELINE_KEY_SCALER_ROBUST, RobustScaler()),
                (EW_PIPELINE_KEY_SCALER_MINMAX, MinMaxScaler())
            ])

        if dimrs is not None:
            params.extend([
                (EW_PIPELINE_KEY_DIMRS, FeatureUnion(dimrs)),
                (EW_PIPELINE_KEY_FEATURES, SelectKBest(k=10))
            ])

        params.append((EW_PIPELINE_KEY_MODEL, model))

        pipeline = Pipeline(params)

        return pipeline

    def _grid(self):
        params = {}

        hyper = self._params[EW_KEY_CONFIG_HYPER] or {}

        self._params[EW_KEY_CONFIG_RGS] = hyper.get('__randomized__', False)

        for key in (EW_KEY_CONFIG_MODEL,):
            params_ = hyper.get(key, {})

            if key == EW_KEY_CONFIG_MODEL:
                self._params[EW_KEY_CONFIG_RGS] = params_.get('__randomized__', False)

            params_ = filter(lambda x: x[0][:2] != '__', params_.items())

            for param, values in params_:
                name = f'{key}__{param}'

                params[name] = values

        grid = params

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

        if self._params[EW_KEY_CONFIG_RGS] is True:
            clf = RandomizedSearchCV(pipeline, param_distributions=grid, cv=inner, iid=iid)
            clf.fit(X, y)
        else:
            clf = GridSearchCV(pipeline, param_grid=[grid], cv=inner, iid=iid)
            clf.fit(X, y)

        if self._eparams[EW_KEY_EVAL_NESTED] is True:
            outer = self._splitter(EW_KEY_CONFIG_OUTER)
            scores = cross_val_score(clf, X=X, y=y, cv=outer)
            score = scores.mean()
        else:
            scores = []
            score = clf.best_score_

        return {
            'results': clf.cv_results_,
            'winner': {
                'estimator': clf.best_estimator_,
                'score': clf.best_score_,
                'params': clf.best_params_,
                'index': clf.best_index_
            },
            'scorer': clf.scorer_
            'scores': scores
        }

    def __init__(self):
        self._params = {
            EW_KEY_CONFIG_DIMRS: None,
            EW_KEY_CONFIG_MODEL: None,
            EW_KEY_CONFIG_HYPER: None,
            EW_KEY_CONFIG_INNER: None,
            EW_KEY_CONFIG_OUTER: None,
            EW_KEY_CONFIG_RGS: None,
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

        model, args, kwargs = self._params[EW_KEY_CONFIG_MODEL]
        self._eparams[EW_KEY_EVAL_MODEL] = model(*args, **kwargs)

        trials = self._params[EW_KEY_CONFIG_TRIALS]
        scores = np.zeros((trials, 1), dtype=np.float64)

        dimrs = self._dimrs()
        self._eparams[EW_KEY_EVAL_DIMRS] = dimrs

        pipeline = self._pipeline()
        grid = self._grid()

        for i in range(trials):
            scores[i, :] = self._trial(pipeline, grid)

        self._reset()

        return scores

    @property
    def dimrs(self):
        return self._params[EW_KEY_CONFIG_MODEL]

    @dimrs.setter
    def dimrs(self, params):
        self._params[EW_KEY_CONFIG_DIMRS] = params

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
