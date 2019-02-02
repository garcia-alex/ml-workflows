import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from utilities import Data, URL_ROOT, PATH_DATASETS
from pipeline import (
    NumericPipeline, CategoryPipeline, UnionPipeline,
    PIPELINE_CATEGORICAL, PIPELINE_NUMERIC, DTYPE_NUMBER
)
from evaluation import Evaluation

URL_HOUSING = f"{URL_ROOT}/datasets/housing/housing.tgz"
PATH_HOUSING = os.path.join(PATH_DATASETS, "housing")
FILE_HOUSING = 'housing.csv'

KEY_LONGITUDE = 'longitude'
KEY_LATITUDE = 'latitude'
KEY_MEDIAN_AGE = 'housing_median_age'
KEY_TOTAL_ROOMS = 'total_rooms'
KEY_TOTAL_BEDROOMS = 'total_bedrooms'
KEY_POPULATION = 'population'
KEY_HOUSEHOLDS = 'households'
KEY_MEDIAN_INCOME = 'median_income'
KEY_MEDIAN_VALUE = 'median_house_value'
KEY_OCEAN_PROXIMITY = 'ocean_proximity'

KEY_CAT_INCOME = 'cat/income'
KEY_ROOMS_PER_HOUSEHOLD = 'rooms/hh'
KEY_BEDROOMS_PER_ROOM = 'bedrooms/room'


def load_housing_data():
    tgz = Data.fetch(URL_HOUSING, PATH_HOUSING)
    Data.extract(tgz, PATH_HOUSING)
    df = Data.load(PATH_HOUSING, FILE_HOUSING)

    return df


def generate_sets(df):
    df[KEY_CAT_INCOME] = np.ceil(df[KEY_MEDIAN_INCOME] / 1.5)
    df[KEY_CAT_INCOME].where(df[KEY_CAT_INCOME] < 5, 5.0, inplace=True)

    train, test = Data.split_strat(df, KEY_CAT_INCOME, 0.2)

    for set_ in (df, train, test):
        set_.drop(KEY_CAT_INCOME, axis=1, inplace=True)

    return train, test


def agents(df):
    keys = [KEY_HOUSEHOLDS, KEY_TOTAL_BEDROOMS, KEY_TOTAL_ROOMS]
    columns = list(df.columns)
    index = dict([(key, columns.index(key)) for key in keys])

    def _agent_rphh(X):
        x = X[:, index[KEY_TOTAL_ROOMS]] / X[:, index[KEY_HOUSEHOLDS]]
        x = np.where(x > 2, x, 2.0)
        x = np.where(x < 8, x, 8.0)

        return x

    def _agent_brpr(X):
        x = X[:, index[KEY_TOTAL_BEDROOMS]] / X[:, index[KEY_TOTAL_ROOMS]]
        x = np.where(x > 0.1, x, 0.1)
        x = np.where(x < 0.4, x, 0.4)

        return x

    return [_agent_rphh, _agent_brpr]


if __name__ == '__main__':
    df = load_housing_data()

    train, test = generate_sets(df)
    labels = train[KEY_MEDIAN_VALUE].copy()

    prep = train.copy()

    pipelines = [
        (PIPELINE_NUMERIC, NumericPipeline(prep, agents(prep))),
        (PIPELINE_CATEGORICAL, CategoryPipeline(prep))
    ]

    pipeline = UnionPipeline(pipelines)

    array = pipeline.fit_transform(prep)

    columns = (
        list(df.select_dtypes(include=[DTYPE_NUMBER]).columns) + [
            KEY_ROOMS_PER_HOUSEHOLD,
            KEY_BEDROOMS_PER_ROOM
        ] + list(prep[KEY_OCEAN_PROXIMITY].unique())
    )

    prep = pd.DataFrame(array, columns=columns)

    train = prep.drop(KEY_MEDIAN_VALUE, axis=1)

    Evaluation(LinearRegression).run(train, labels)
    Evaluation(DecisionTreeRegressor).run(train, labels)
    Evaluation(RandomForestRegressor, n_estimators=10, random_state=42).run(train, labels)
