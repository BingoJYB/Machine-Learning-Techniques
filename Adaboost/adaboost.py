import numpy as np


def read_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


def decision_stump_one_dim(X, Y):
    min_Ein = np.inf
    best_s = 0
    best_theta = 0
    thresholds = [-np.inf]

    data = sorted(zip(X, Y))
    X = np.asarray(list(zip(*data))[0])
    Y = np.asarray(list(zip(*data))[1])

    for i in range(X.shape[0]-1):
        thresholds.append(X[i] + X[i+1] / 2.0)

    for s in [1, -1]:
        for theta in thresholds:
            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = 1 * s
            temp_X[temp_X < theta] = -1 * s
            Ein1 = np.sum(temp_X != Y) / X.shape[0]

            temp_X = np.copy(X)
            temp_X[temp_X >= theta] = -1 * s
            temp_X[temp_X < theta] = 1 * s
            Ein2 = np.sum(temp_X != Y) / X.shape[0]

            Ein = min(Ein1, Ein2)
            if Ein < min_Ein:
                min_Ein = Ein
                best_s = s
                best_theta = theta

    return min_Ein, best_s, best_theta


def decision_stump_multi_dim(X, Y):
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
        Ein, s, theta = decision_stump_one_dim(x, Y)
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


if __name__ == '__main__':
    train_X, train_Y = read_data('hw2_adaboost_train.dat')
    test_X, test_Y = read_data('hw2_adaboost_test.dat')

    min_Ein, best_s, best_theta, best_dim = decision_stump_multi_dim(train_X, train_Y)

    print(min_Ein, best_s, best_theta, best_dim)
