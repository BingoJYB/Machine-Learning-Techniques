import numpy as np


def calculateOneDimEin(X, Y):
    min_Ein = np.inf
    best_s = 0
    best_theta = 0

    for s in [1, -1]:
        for theta in np.nditer(X):
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
                best_theta = abs(theta)

    return min_Ein, best_s, best_theta


def calculateMultiDimEin(X, Y):
    Eins = []
    ss = []
    thetas = []
    min_Ein = np.inf
    best_s = 0
    best_theta = 0
    best_dim = 0
    Y = np.asarray(Y, dtype=float)

    for x in X:
        x = np.asarray(x, dtype=float)
        Ein, s, theta = calculateOneDimEin(x, Y)
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

