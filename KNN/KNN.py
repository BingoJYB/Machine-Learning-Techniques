import numpy as np


def read_data(filename):
    data = np.loadtxt(filename)

    return data


def gaussian_rbf(gamma, x1, x2):
    delta = x1 - x2

    return np.exp(-gamma * np.dot(delta, delta.T))


def get_k_nearest_neighbor(gamma, k, element, data):
    k_nearest_distance = [np.inf] * k
    k_nearest_neighbor = [element] * k

    for candidate in data:
        distance = gaussian_rbf(gamma, element[:, :-1], candidate[:, :-1])

        if distance < max(k_nearest_distance):
            idx = k_nearest_distance.index(max(k_nearest_distance))
            k_nearest_distance[idx] = distance
            k_nearest_neighbor[idx] = candidate[:, -1]

    return k_nearest_neighbor


def predict(gamma, k, element, data):
    k_nearest_neighbor = get_k_nearest_neighbor(gamma, k, element, data)

    return np.sign(sum(k_nearest_neighbor))


def get_err(data, prediction):

    return


if __name__ == '__main__':
    train_data = read_data('hw8_train.dat')
    test_data = read_data('hw8_test.dat')

