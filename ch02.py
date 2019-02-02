import os

import numpy as np

from util import (
    URL_ROOT, PATH_DATASETS,
    Data
)


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


def engineer_features(df):
    key = KEY_ROOMS_PER_HOUSEHOLD
    df[key] = df[KEY_TOTAL_ROOMS] / df[KEY_HOUSEHOLDS]
    df[key].where(df[key] > 2, 2.0, inplace=True)
    df[key].where(df[key] < 8, 8.0, inplace=True)

    key = KEY_BEDROOMS_PER_ROOM
    df[key] = df[KEY_TOTAL_BEDROOMS] / df[KEY_TOTAL_ROOMS]
    df[key].where(df[key] < 0.4, 0.4, inplace=True)
    df[key].where(df[key] > 0.1, 0.1, inplace=True)


if __name__ == '__main__':
    df = load_housing_data()

    prep = df.copy()
    engineer_features(prep)

    prep, medians = Data.impute(prep)
    prep, categories = Data.onehot(prep, KEY_OCEAN_PROXIMITY)

    train, test = generate_sets(prep)

    labels = train[KEY_MEDIAN_VALUE].copy()
    train = train.drop(KEY_MEDIAN_VALUE, axis=1)
