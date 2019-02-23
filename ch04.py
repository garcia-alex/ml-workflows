import math
import functools

import numpy as np
from sklearn import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.linear_model import LinearRegression


METHOD_NE = 'normal_equation'
METHOD_SVD = 'svd_decomposition'
METHOD_BGD = 'batch_gradient_descent'
METHOD_SGD = 'stochastic_gradient_descent'
METHOD_MBGD = 'mini_batch_gradient_descent'


def bias(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        has_bias = kwargs.get('has_bias', False)
        X = args[0]

        if has_bias:
            X_ = X
        else:
            X_ = add_bias_vector(X)

        args_ = [X_] + list(args[1:])

        result = method(*args_, **kwargs)

        return result
    return wrapper


def add_bias_vector(X):
    shape = (len(X), 1)
    o = np.ones(shape)

    X_ = np.c_[o, X]

    return X_


@bias
def normal_equation(X, y, alpha=0):
    A = np.identity(len(X[0]))
    A[0][0] = 0

    theta = np.linalg.inv(X.T.dot(X) + alpha*A)
    theta = theta.dot(X.T).dot(y)

    return theta


@bias
def svd_decomposition(X, y):
    theta = np.linalg.pinv(X).dot(y)

    return theta


@bias
def batch_gradient_descent(X, y, lr=0.1, niter=1000, tolerance=0.0001):
    m = len(X)
    theta = np.random.randn(len(X[0]), 1)
    path = []

    for _ in range(niter):
        grads = 2 / m * X.T.dot(X.dot(theta) - y)

        incrs = lr * grads
        path.append(theta)
        theta = theta - incrs

        if np.linalg.norm(grads) < tolerance:
            break

    path.append(theta)

    return theta, np.array(path)


@bias
def stochastic_gradient_descent(X, y, epochs=50, lr=0.1, drop=0.5, cycle=10):
    ilr = lr
    lr = lambda x: ilr * pow(drop, math.floor(x / cycle))

    m = len(X)
    theta = np.random.randn(len(X[0]), 1)
    path = []

    for epoch in range(epochs):
        for _ in range(m):
            j = np.random.randint(m)

            x_ = X[j: j + 1]
            y_ = y[j: j + 1]

            grads = 2 * x_.T.dot(x_.dot(theta) - y_)

            incrs = lr(epoch) * grads
            path.append(theta)
            theta = theta - incrs

    path.append(theta)

    return theta, np.array(path)


@bias
def mini_batch_gradient_descent(X, y, lr=0.1, batch=10, niter=1000, tolerance=0.0001):
    m = len(X)
    theta = np.random.randn(len(X[0]), 1)
    path = []

    for _ in range(niter):
        index = np.random.choice(m, batch, replace=True)

        x_ = X[index]
        y_ = y[index]

        grads = 2 / batch * x_.T.dot(x_.dot(theta) - y_)

        incrs = lr * grads
        path.append(theta)
        theta = theta - incrs

        if np.linalg.norm(grads) < tolerance:
            break

    path.append(theta)

    return theta, np.array(path)


def generate_data(theta, count=100, v=2, u=0):
    np.random.seed(seed=42)

    X = v * np.random.rand(count, 1) + u
    noise = np.random.randn(count, 1)

    poly = sum([n * pow(X, i) for (i, n) in enumerate(theta)])

    y = poly + noise

    return (X, y)


def linear_fit(X, y, algo=METHOD_NE, *args, **kwargs):
    mapping = {
        METHOD_NE: normal_equation,
        METHOD_SVD: svd_decomposition,
        METHOD_BGD: batch_gradient_descent,
        METHOD_SGD: stochastic_gradient_descent,
        METHOD_MBGD: mini_batch_gradient_descent
    }

    method = mapping[algo]

    theta = method(X, y, *args, **kwargs)

    return theta


@bias
def linear_predict(X, theta):
    return X.dot(theta)


def learning_curves(X, y, n=5, rs=None, aggr='mean'):
    def one_pass():
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=rs)
        model = LinearRegression()

        def fit_predict_subset(m):
            estimator = clone(model)
            estimator.fit(Xt[:m], yt[:m])
            ypt = estimator.predict(Xt[:m])
            ypv = estimator.predict(Xv)
            et = mean_squared_error(yt[:m], ypt)
            ev = mean_squared_error(yv, ypv)
            return (et, ev)

        lengths = range(1, len(Xt))
        errors = map(fit_predict_subset, lengths)
        errors = np.array(list(errors))
        rmset = np.sqrt(errors[:, 0])
        rmsev = np.sqrt(errors[:, 1])

        return (rmset, rmsev)

    Et, Ev = [], []

    for i in range(n):
        et, ev = one_pass()
        Et.append(et)
        Ev.append(ev)

    method = getattr(np, aggr)
    et = method(np.array(list(zip(*Et))), axis=1)
    ev = method(np.array(list(zip(*Ev))), axis=1)

    return (et, ev)


def learning_curves_2(X, y):
    model = LinearRegression()
    m = int(len(X) * 0.8)

    _, Et, Ev = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=range(1, m),
        cv=5,
        scoring='neg_mean_squared_error'
    )

    mset = -Et.mean(axis=1)
    msev = -Ev.mean(axis=1)

    rmset = np.sqrt(mset)
    rmsev = np.sqrt(msev)

    return (rmset, rmsev)


if __name__ == '__main__':
    np.set_printoptions(precision=4)

    theta = [4, 3]
    X, y = generate_data(theta)
