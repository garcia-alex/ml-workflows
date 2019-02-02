import os
from six.moves import urllib
import tarfile
from zlib import crc32

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


PATH_DATASETS = "datasets"
URL_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"

RANDOM_STATE = 42


class Data(object):
    @staticmethod
    def fetch(url, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        fname = os.path.split(url)[-1]
        fname = os.path.join(path, fname)

        urllib.request.urlretrieve(url, fname)

        return fname

    @staticmethod
    def extract(tgz, path):
        tgz = tarfile.open(tgz)
        tgz.extractall(path=path)
        tgz.close()

    @staticmethod
    def load(path, fname):
        csv_path = os.path.join(path, fname)
        df = pd.read_csv(csv_path)

        return df

    @staticmethod
    def split(df, index, rt=0.2, rv=0.0):
        d = 2 ** 32

        def crc(id):
            c = crc32(np.int64(id)) & 0xffffffff
            return c

        def is_t(id):
            c = crc(id)
            return c < rt * d

        def is_v(id):
            c = crc(id)
            return c >= rt * d and c < (rt + rv) * d

        def is_m(id):
            c = crc(id)
            return c >= (rt + rv) * d

        ids = df[index]
        it = ids.apply(is_t)
        iv = ids.apply(is_v)
        im = ids.apply(is_m)

        return df.loc[im], df.loc[it], df.loc[iv]

    @staticmethod
    def split_strat(df, key, test_size):
        split = SSS(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        itrain, itest = tuple(split.split(df, df[key]))[0]

        train = df.loc[itrain]
        test = df.loc[itest]

        return train, test

    @staticmethod
    def impute(df, strategy='median'):
        numeric = df.select_dtypes(include=['number'])

        imputer = SimpleImputer(strategy=strategy)

        # consider fit_transform instead
        imputer.fit(numeric)
        imputed = imputer.transform(numeric)
        imputed = pd.DataFrame(imputed, columns=numeric.columns)

        columns = df.select_dtypes(exclude=['number']).columns
        imputed[columns] = df[columns]

        params = imputer.statistics_

        return (imputed, params)

    @staticmethod
    def onehot(df, key):
        # consider scikit.CategoricalEncoder() instrad
        encoder = OneHotEncoder(categories='auto')

        encoded, categories = df[key].fillna('NULL').factorize()

        sparse = encoder.fit_transform(encoded.reshape(-1, 1))
        encoded = sparse.toarray()

        encoded = pd.DataFrame(encoded)
        encoded.columns = categories

        df = df.drop(key, axis=1)
        encoded[df.columns] = df

        return (encoded, categories)
