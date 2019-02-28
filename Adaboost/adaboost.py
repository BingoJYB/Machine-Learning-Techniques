import numpy as np


def read_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


def decision_stump_one_dim(U, X, Y):
    min_err = 1
    best_s = 0
    best_theta = 0
    thetas = [-np.inf]
    X_hat = np.sort(X)

    for i in range(len(X_hat) - 1):
        thetas.append((X_hat[i] + X_hat[i + 1]) / 2)

    for theta in thetas:
        s = 1
        y_predict = np.sign(X - theta)
        err = np.sum((Y != y_predict) * U) / len(X)
        err2 = np.sum((Y != (-1 * y_predict)) * U) / len(X)

        if err2 < err:
            err = err2
            s = -1

        if err <= min_err:
            best_s = s
            best_theta = theta
            min_err = err

    return min_err, best_s, best_theta


def decision_stump_multi_dim(U, X, Y):
    min_err = 1
    best_s = 0
    best_theta = 0
    best_dim = 0

    for col in range(X.shape[1]):
        err, s, theta = decision_stump_one_dim(U, X[:, col], Y)

        if err < min_err:
            min_err = err
            best_s = s
            best_theta = theta
            best_dim = col

    return min_err, best_s, best_theta, best_dim


def gt(s, theta, X):
    return s * np.sign(X - theta)


def update_parameter(U, X, Y, s, theta, err):
    epsilon = err / np.sum(U)
    t = np.sqrt((1 - epsilon) / epsilon)
    Y_hat = gt(s, theta, X)
    U = np.where(Y_hat != Y, U * t, U / t)
    alpha = np.log(t)

    return U, alpha


if __name__ == '__main__':
    train_X, train_Y = read_data('hw2_adaboost_train.dat')
    test_X, test_Y = read_data('hw2_adaboost_test.dat')
    U = 1 / train_X.shape[0] * np.ones(train_X.shape[0])
    G = []
    err_in_G = np.zeros(test_X.shape[0])

    for i in range(300):
        err, s, theta, dim = decision_stump_multi_dim(U, train_X, train_Y)
        U, alpha = update_parameter(U, train_X[:, dim], train_Y, s, theta, err)
        G.append((alpha, s, theta, dim))

    for g_param in G:
        err_in_G = err_in_G + g_param[0] * gt(g_param[1], g_param[2], test_X[:, g_param[3]])

    print(np.sum(np.sign(err_in_G) != test_Y) / test_X.shape[0])
