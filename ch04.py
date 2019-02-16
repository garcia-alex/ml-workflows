import math
import numpy as np


METHOD_NE = 'normal_equation'
METHOD_SVD = 'svd_decomposition'
METHOD_BGD = 'batch_gradient_descent'
METHOD_SGD = 'stochastic_gradient_descent'


def add_bias_vector(X):
    shape = (len(X), 1)
    o = np.ones(shape)

    X_ = np.c_[o, X]

    return X_


def normal_equation(X, y):
    X_ = add_bias_vector(X)

    theta = np.linalg.inv(X_.T.dot(X_))
    theta = theta.dot(X_.T).dot(y)

    return theta


def svd_decomposition(X, y):
    X_ = add_bias_vector(X)

    theta = np.linalg.pinv(X_).dot(y)

    return theta


def batch_gradient_descent(X, y, lr=0.1, niter=1000, tolerance=0.0001):
    X_ = add_bias_vector(X)
    m = len(X)
    theta = np.random.randn(len(X_[0]), 1)

    for i in range(niter):
        grads = 2 / m * X_.T.dot(X_.dot(theta) - y)

        incrs = lr * grads
        theta = theta - incrs

        if np.linalg.norm(grads) < tolerance:
            print(i)
            break

    return theta


def stochastic_gradient_descent(X, y, epochs=50, lr=0.1, drop=0.5, cycle=10):
    ilr = lr
    lr = lambda x: ilr * pow(drop, math.floor(x / cycle))

    X_ = add_bias_vector(X)
    m = len(X)
    theta = np.random.randn(len(X_[0]), 1)

    for epoch in range(epochs):
        for i in range(m):
            j = np.random.randint(m)

            x_ = X_[j: j + 1]
            y_ = y[j: j + 1]

            grads = 2 * x_.T.dot(x_.dot(theta) - y_)

            incrs = lr(epoch) * grads
            theta = theta - incrs

    return theta


def generate_data(theta, count=100):
    np.random.seed(seed=42)

    X = 2 * np.random.rand(count, 1)
    noise = np.random.randn(count, 1)

    poly = sum([n * pow(X, i) for (i, n) in enumerate(theta)])

    y = poly + noise

    return (X, y)


def linear_fit(X, y, algo=METHOD_NE, *args, **kwargs):
    mapping = {
        METHOD_NE: normal_equation,
        METHOD_SVD: svd_decomposition,
        METHOD_BGD: batch_gradient_descent,
        METHOD_SGD: stochastic_gradient_descent
    }

    method = mapping[algo]

    theta = method(X, y, *args, **kwargs)

    return theta


def linear_predict(X, theta):
    shape = (len(X), 1)
    o = np.ones(shape)

    X_ = np.c_[o, X]

    y = X_.dot(theta)

    return y


if __name__ == '__main__':
    np.set_printoptions(precision=4)

    theta = [4, 3]
    X, y = generate_data(theta)
