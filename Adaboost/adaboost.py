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

    data = sorted(zip(X, Y, U))
    X = np.asarray(list(zip(*data))[0])
    Y = np.asarray(list(zip(*data))[1])
    U = np.asarray(list(zip(*data))[2])

    for i in range(X.shape[0]-1):
        thresholds.append(X[i] + X[i+1] / 2.0)

    for s in [1, -1]:
        for theta in thresholds:
            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = 1 * s
            temp_X[temp_X < theta] = -1 * s
            Ein1 = np.sum((temp_X != Y) * U)

            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = -1 * s
            temp_X[temp_X < theta] = 1 * s
            Ein2 = np.sum((temp_X != Y) * U)

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


def update_parameter(U, X, Y, s, theta, Ein):
    epsilon = Ein / np.sum(U)
    diamond_t = np.sqrt((1 - epsilon) / epsilon)

    Y_prime = gt(s, theta, X)
    incorrect = (Y_prime != Y) * U * diamond_t
    correct = (Y_prime == Y) * U / diamond_t
    new_U = correct + incorrect
    alpha = np.log(diamond_t)

    return new_U, alpha


if __name__ == '__main__':
    train_X, train_Y = read_data('hw2_adaboost_train.dat')
    test_X, test_Y = read_data('hw2_adaboost_test.dat')
    U = 1 / train_X.shape[0] * np.ones(train_X.shape[0])
    gt_parameters = []
    result = 0

    for i in range(300):
        Ein, s, theta, dim = decision_stump_multi_dim(U, train_X, train_Y)
        U, alpha = update_parameter(U, train_X[:, dim], train_Y, s, theta, Ein)
        gt_parameters.append((alpha, s, theta, dim))

    for parameters in gt_parameters:
        result = result + parameters[0] * gt(parameters[1], parameters[2], train_X[:, parameters[3]])

    print(np.sign(result))
