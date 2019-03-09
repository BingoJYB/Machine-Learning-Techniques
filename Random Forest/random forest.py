import numpy as np
from dtree import build_tree, predict


def read_data(filename):
    data = np.loadtxt(filename)

    return data


def bagging(data):
    row, col = data.shape
    pos = np.random.randint(0, row, (row,))

    return data[pos]


def random_forest(data, T):
    forest = []

    for i in range(T):
        data_temp = bagging(data)
        tree = build_tree(data_temp, 5, 1)
        forest.append(tree)

    return forest


if __name__ == '__main__':
    train_data = read_data('hw7_train.dat')
    test_data = read_data('hw7_test.dat')
    ein = 0
    eout = 0

    for j in range(50):
        forest = random_forest(train_data, 300)
        size = len(forest)
        yhat1 = np.zeros((train_data.shape[0], size))
        yhat2 = np.zeros((test_data.shape[0], size))

        for i in range(size):
            train_predictions = list()
            test_predictions = list()

            for row in train_data:
                train_prediction = predict(forest[i], row)
                train_predictions.append(train_prediction)

            for row in test_data:
                test_prediction = predict(forest[i], row)
                test_predictions.append(test_prediction)

            yhat1[:, i:i + 1] = np.asarray(train_predictions).reshape(train_data.shape[0], 1)
            yhat2[:, i:i + 1] = np.asarray(test_predictions).reshape(test_data.shape[0], 1)

        Yhat = np.sign(np.sum(yhat1, 1)).reshape(1, train_data.shape[0])
        Ytesthat = np.sign(np.sum(yhat2, 1)).reshape(1, test_data.shape[0])
        Yhat[Yhat == 0] = 1
        Ytesthat[Ytesthat == 0] = 1
        ein += np.sum(Yhat != train_data[:, -1]) / train_data.shape[0]
        eout += np.sum(Ytesthat != test_data[:, -1]) / test_data.shape[0]

    print('Ein(G): ', ein / 50)
    print('Eout(G): ', eout / 50)
