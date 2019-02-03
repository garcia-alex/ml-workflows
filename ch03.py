import os
import numpy as np

from utilities import Data, PATH_DATASETS

MNIST_PATH = os.path.join(PATH_DATASETS, "mnist")
MNIST_DATASET = 'mnist_784'
MNIST_KEY_DATA = 'data'
MNIST_KEY_TARGET = 'target'


def fetch_mnist():
    mnist = Data.fetch_openml(
        MNIST_PATH, MNIST_DATASET,
        version=1, cache=True
    )

    return mnist


def process_mnist(mnist):
    n = 60000

    mnist.target = mnist.target.astype(np.int8)

    train = mnist.target[:n]
    test = mnist.target[n:]

    train = sorted(zip(train, range(len(train))))
    test = sorted(zip(test, range(len(test))))

    train = np.array(train)[:, 1]
    test = np.array(test)[:, 1]

    mnist.data[:n] = mnist.data[train]
    mnist.target[:n] = mnist.target[train]

    mnist.data[n:] = mnist.data[test + n]
    mnist.target[n:] = mnist.target[test + n]


if __name__ == '__main__':
    mnist = fetch_mnist()
    process_mnist(mnist)

    X, y = mnist[MNIST_KEY_DATA], mnist[MNIST_KEY_TARGET]
