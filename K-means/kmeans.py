import numpy as np


def read_data(filename):
    data = np.loadtxt(filename)

    return data


def initialize(k, data):
    initial_value_index = np.random.randint(data.shape[0], size=k)

    return data[initial_value_index]


def get_euclidean(x1, x2):
    delta = x1 - x2

    return np.dot(delta, delta.T)


def find_cluster(cluster_center, data):
    new_clusters = {key: [] for key in range(len(cluster_center))}

    for element in data:
        min_distance = np.inf
        cluster_id = -1

        for id, center in enumerate(cluster_center):
            distance = get_euclidean(element, center)

            if distance < min_distance:
                min_distance = distance
                cluster_id = id

        new_clusters[cluster_id].append(element)

    return new_clusters


def calculate_center(clusters):
    new_cluster_center = []

    for cluster in clusters.values():
        new_cluster_center.append(np.mean(cluster, axis=0))

    return new_cluster_center


def Kmeans(data, k, iter_number):
    cluster_center = initialize(k, data)
    clusters = {key: [] for key in range(k)}

    for i in range(iter_number):
        clusters = find_cluster(cluster_center, data)
        cluster_center = calculate_center(clusters)

    return cluster_center, clusters


def get_err(cluster_center, clusters):
    err = 0

    for id, cluster in clusters.items():
        delta = cluster - cluster_center[id]
        mat = np.dot(delta, delta.T)
        diagonal_in_mat = mat[np.diag_indices_from(mat)]
        err += np.sum(diagonal_in_mat)

    return err


if __name__ == '__main__':
    train_data = read_data('hw8_nolabel_train.dat')
    cluster_center, clusters = Kmeans(train_data, 10, 500)
    print(get_err(cluster_center, clusters) / train_data.shape[0])
