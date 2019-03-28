import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, KBinsDiscretizer


DTYPE_NUMBER = 'number'
DTYPE_OBJECT = 'object'

NAME_PERMUTATOR = 'permutator'
NAME_SELECTOR = 'selector'
NAME_IMPUTER = 'imputer'
NAME_ENGINEER = 'engineer'
NAME_SCALER_STD = 'scaler_std'
NAME_SCALER_NORM = 'scaler_norm'
NAME_DISCRETIZER = 'discretizer'
NAME_FEATURES_POLY = 'features_poly'
NAME_ENCODER = 'encoder'
NAME_MODEL = 'model'

PIPELINE_NUMERIC = 'numeric'
PIPELINE_CATEGORICAL = 'categorical'

STRATEGY_MEDIAN = 'median'
STRATEGY_CONSTANT = 'constant'


class Permutator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X, y


class AttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if type(X) != pd.core.frame.DataFrame:
            X = pd.DataFrame(X)

        return X[self.attributes].values


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, agents):
        self.agents = agents

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X[:, :]

        for agent in self.agents:
            array = agent(X_)
            X_ = np.c_[X_, array]

        return X_


class DefaultPipeline(Pipeline):
    def __init__(self):
        pipeline = [
            (NAME_PERMUTATOR, Permutator()),
        ]

        super(DefaultPipeline, self).__init__(pipeline)


class NumericPipeline(Pipeline):
    def __init__(self, X, agents=[], strategy=STRATEGY_MEDIAN, scale=True, discrete=False, degree=1, bias=False, model_=None):
        if type(X) != pd.core.frame.DataFrame:
            X = pd.DataFrame(X)

        self.attributes = X.select_dtypes(include=[DTYPE_NUMBER]).columns

        pipeline = [
            (NAME_SELECTOR, AttributeSelector(self.attributes)),
            (NAME_IMPUTER, SimpleImputer(strategy=strategy)),
            (NAME_ENGINEER, FeatureEngineer(agents))
        ]

        if scale is True:
            pipeline.append(
                (NAME_SCALER_STD, StandardScaler())
            )

        if discrete is True:
            pipeline.append(
                (NAME_DISCRETIZER, KBinsDiscretizer(n_bins=bins))
            )

        if bias is True or degree > 1:
            poly = PolynomialFeatures(degree=degree, include_bias=bias)
            pipeline.append(
                (NAME_FEATURES_POLY, poly)
            )

        if model_ is not None:
            pipeline.append(
                (NAME_MODEL, model_)
            )

        super(NumericPipeline, self).__init__(pipeline)


class CategoryPipeline(Pipeline):
    def __init__(self, X, sparse=False):
        if type(X) != pd.core.frame.DataFrame:
            X = pd.DataFrame(X)

        self.attributes = X.select_dtypes(exclude=[DTYPE_NUMBER]).columns
        encoder = OneHotEncoder() if oh else LabelEncoder()

        pipeline = [
            (NAME_SELECTOR, AttributeSelector(self.attributes)),
            (NAME_IMPUTER, SimpleImputer(strategy=STRATEGY_CONSTANT, fill_value=' ')),
            (NAME_ENCODER, OneHotEncoder(sparse=sparse))
        ]

        super(CategoryPipeline, self).__init__(pipeline)


class UnionPipeline(FeatureUnion):
    def __init__(self, pipelines):
        super(UnionPipeline, self).__init__(transformer_list=pipelines)
