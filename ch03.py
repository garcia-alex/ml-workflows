import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

from const import PATH_DATASETS
from utilities import Data

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


def shuffle_mnist(X, y):
    index = np.random.permutation(len(X))
    X_ = X[index]
    y_ = y[index]

    return X_, y_


def visualize_mnist(features, i):
    digit = features[i]
    image = digit.reshape(28, 28)

    plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


def visualize_mnist_multi(instances, images_per_row=10, **options):
    size = 28

    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1
    
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    
    image = np.concatenate(row_images, axis=0)
    
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


def prc_graph(p, r, t, show=True):
    if show:
        plt.figure(figsize=(8, 4))
    plt.plot(t, p, 'b--', label="Precision")
    plt.plot(t, r, 'g-', label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.xlim([-700000, 700000])
    if show:
        plt.show()


def pr_graph(p, r, show=True):
    if show:
        plt.figure(figsize=(8, 4))
    plt.plot(r, p, 'b-', label="Precision")
    plt.xlabel("Recall")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    if show:
        plt.show()


def roc_graph(fpr, tpr, label=None, show=True):
    if show:
        plt.figure(figsize=(8, 4))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    if show:
        plt.show()


if __name__ == '__main__':
    mnist = fetch_mnist()
    extract_mnist(mnist)

    n = MNIST_LEN_TRAIN

    train = (mnist[MNIST_KEY_DATA][:n], mnist[MNIST_KEY_TARGET][:n])
    test = (mnist[MNIST_KEY_DATA][n:], mnist[MNIST_KEY_TARGET][n:])

    shuffled = shuffle_mnist(*train)
    features, labels = shuffled
    labels_binary = (labels == 5)
