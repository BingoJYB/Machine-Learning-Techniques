import numpy as np


def read_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


def decision_stump_one_dim(U, X, Y):
    min_Ein = np.inf
    best_s = 0
    best_theta = 0
    thresholds = [-np.inf]
    X_hat = np.asarray(sorted(X))

    for i in range(X_hat.shape[0]-1):
        thresholds.append(X_hat[i] + X_hat[i+1] / 2)

    for s in [1, -1]:
        for theta in thresholds:
            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = 1 * s
            temp_X[temp_X < theta] = -1 * s
            Ein1 = (temp_X != Y).T.dot(U)

            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = -1 * s
            temp_X[temp_X < theta] = 1 * s
            Ein2 = (temp_X != Y).T.dot(U)

            Ein = min(Ein1, Ein2)
            if Ein < min_Ein:
                min_Ein = Ein
                best_s = s
                best_theta = theta

    return min_Ein, best_s, best_theta


def decision_stump_multi_dim(U, X, Y):
    Eins = []
    ss = []
    thetas = []
    min_Ein = np.inf
    best_s = 0
    best_theta = 0
    best_dim = 0
    Y = np.asarray(Y, dtype=float)

    for col in range(X.shape[1]):
        x = np.asarray(X[:, col], dtype=float)
        Ein, s, theta = decision_stump_one_dim(U, x, Y)
        Eins.append(Ein)
        ss.append(s)
        thetas.append(theta)

    for idx, val in enumerate(Eins):
        if val < min_Ein:
            min_Ein = val
            best_s = ss[idx]
            best_theta = thetas[idx]
            best_dim = idx

    return min_Ein, best_s, best_theta, best_dim


def gt(s, theta, X):
    return s * np.sign(X - theta)


def update_parameter(U, X, Y, s, theta, err):
    Y_hat = gt(s, theta, X)
    epsilon = err / np.sum(U)
    diamond_t = np.sqrt((1 - epsilon) / epsilon)
    U = np.where(Y_hat != Y, U * diamond_t, U / diamond_t)
    alpha = np.log(diamond_t)

    return U, alpha


if __name__ == '__main__':
    train_X, train_Y = read_data('hw2_adaboost_train.dat')
    test_X, test_Y = read_data('hw2_adaboost_test.dat')
    U = 1 / train_X.shape[0] * np.ones(train_X.shape[0])
    gt_parameters = []
    result = np.zeros(train_X.shape[0])

    for i in range(300):
        err, s, theta, dim = decision_stump_multi_dim(U, train_X, train_Y)
        U, alpha = update_parameter(U, train_X[:, dim], train_Y, s, theta, err)
        gt_parameters.append((alpha, s, theta, dim))

    for parameters in gt_parameters:
        result = result + parameters[0] * gt(parameters[1], parameters[2], train_X[:, parameters[3]])

    print(np.sum(np.sign(result) != train_Y) / train_X.shape[0])
