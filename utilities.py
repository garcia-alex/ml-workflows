import os
import pickle
import tarfile
from six.moves import urllib
from zlib import crc32

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit as SSS


PATH_DATASETS = "datasets"
URL_HANDSON_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"

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
    def fetch_openml(path, dataset, *args, **kwargs):
        if not os.path.isdir(path):
            os.makedirs(path)

        path = f'{path}/{dataset}'

        if os.path.exists(path):
            f = open(path, 'rb')
            data = pickle.load(f)
            f.close()

        else:
            data = fetch_openml(dataset, *args, **kwargs)
            f = open(path, 'wb')
            pickle.dump(data, f)
            f.close()

        return data

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
