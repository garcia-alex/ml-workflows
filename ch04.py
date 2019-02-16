import numpy as np


def generate_data(theta, count=100):
    X = 2 * np.random.rand(count, 1)
    noise = np.random.randn(count, 1)

    poly = sum([n * pow(X, i) for (i, n) in enumerate(theta)])

    y = poly + noise

    return (X, y)


def linear_fit(X, y):
    shape = (len(X), 1)
    o = np.ones(shape)

    X_ = np.c_[o, X]

    theta = np.linalg.inv(X_.T.dot(X_))
    theta = theta.dot(X_.T).dot(y) 

    return theta


def linear_predict(X, theta):
    shape = (len(X), 1)
    o = np.ones(shape)

    X_ = np.c_[o, X]

    y = X_.dot(theta)

    return y


if __name__ == '__main__':
    formatter = {
        'float': lambda x: f'{x:.4f}'
    }

    np.set_printoptions(formatter=formatter)
    
    theta = [4, 3]
    X, y = generate_data(theta)