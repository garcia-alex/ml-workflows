import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

from utilities import Data, PATH_DATASETS

MNIST_PATH = os.path.join(PATH_DATASETS, "mnist")
MNIST_DATASET = 'mnist_784'
MNIST_KEY_DATA = 'data'
MNIST_KEY_TARGET = 'target'
MNIST_LEN_TRAIN = 60000
MNIST_LEN_TEST = 10000
MNIST_LEN = MNIST_LEN_TRAIN + MNIST_LEN_TEST


def fetch_mnist():
    mnist = Data.fetch_openml(
        MNIST_PATH, MNIST_DATASET,
        version=1, cache=True
    )

    return mnist


def extract_mnist(mnist):
    n = MNIST_LEN_TRAIN

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


def shuffle_mnist_train(X, y):
    index = np.random.permutation(len(X))
    X_ = X[index]
    y_ = y[index]

    return X_, y_


def visualize_mnist(mnist, i):
    X = mnist[0]

    digit = X[i]
    image = digit.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    mnist = fetch_mnist()
    extract_mnist(mnist)

    n = MNIST_LEN_TRAIN

    train = (mnist[MNIST_KEY_DATA][:n], mnist[MNIST_KEY_TARGET][:n])
    test = (mnist[MNIST_KEY_DATA][n:], mnist[MNIST_KEY_TARGET][n:])

    shuffled = shuffle_mnist_train(*train)
    X_, y_ = shuffled

    y_train = (y_ == 5)
    y_test = (test[1] == 5)

    sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
    sgd_clf.fit(X_, y_train)
