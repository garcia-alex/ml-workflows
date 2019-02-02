import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DTYPE_NUMBER = 'number'
DTYPE_OBJECT = 'object'

NAME_SELECTOR = 'selector'
NAME_IMPUTER = 'imputer'
NAME_ENGINEER = 'engineer'
NAME_SCALER_STD = 'scaler_std'
NAME_SCALER_NORM = 'scaler_norm'
NAME_ENCODER = 'encoder'

PIPELINE_NUMERIC = 'numeric'
PIPELINE_CATEGORICAL = 'categorical'

STRATEGY_MEDIAN = 'median'


class AttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
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


class NumericPipeline(Pipeline):
    def __init__(self, df, agents=[], strategy=STRATEGY_MEDIAN, scale=True):
        self.attributes = df.select_dtypes(include=[DTYPE_NUMBER]).columns

        pipeline = [
            (NAME_SELECTOR, AttributeSelector(self.attributes)),
            (NAME_IMPUTER, SimpleImputer(strategy=strategy)),
            (NAME_ENGINEER, FeatureEngineer(agents))
        ]

        if scale:
            pipeline.append(
                (NAME_SCALER_STD, StandardScaler())
            )

        super(NumericPipeline, self).__init__(pipeline)


class CategoryPipeline(Pipeline):
    def __init__(self, df, sparse=False):
        self.attributes = df.select_dtypes(exclude=[DTYPE_NUMBER]).columns

        pipeline = [
            (NAME_SELECTOR, AttributeSelector(self.attributes)),
            (NAME_ENCODER, OneHotEncoder(sparse=sparse))
        ]

        super(CategoryPipeline, self).__init__(pipeline)


class UnionPipeline(FeatureUnion):
    def __init__(self, pipelines):
        super(UnionPipeline, self).__init__(transformer_list=pipelines)
