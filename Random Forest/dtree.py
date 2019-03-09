import numpy as np
from functools import reduce


def read_data(filename):
    data = np.loadtxt(filename)

    return data


def get_gini(groups):
    total_size = reduce(lambda size, group: size + len(group), groups, 0)
    gini = 0

    for group in groups:
        group_size = len(group)
        type_1 = 0
        type_2 = 0

        if group_size == 0:
            continue

        for row in group:
            if row[-1] == 1:
                type_1 = type_1 + 1
            else:
                type_2 = type_2 + 1

        score = 1 - (type_1 / group_size) ** 2 - (type_2 / group_size) ** 2
        gini += score * group_size / total_size

    return gini


def test_split(feature, val, data):
    left, right = list(), list()

    for row in data:
        if row[feature] < val:
            left.append(row)
        else:
            right.append(row)

    return left, right


def get_split(data):
    best_feature, best_val, min_gini, groups = np.inf, np.inf, np.inf, None

    for i in range(len(data[0]) - 1):
        for row in data:
            left, right = test_split(i, row[i], data)

            gini = get_gini([left, right])
            if gini < min_gini:
                best_feature = i
                best_val = row[i]
                min_gini = gini
                groups = [left, right]

    return {'index': best_feature, 'value': best_val, 'groups': groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]

    return max(set(outcomes), key=outcomes.count)


def split(data, max_depth, min_size, depth):
    left, right = data['groups']
    del (data['groups'])

    if not left or not right:
        data['left'] = data['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        data['left'], data['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        data['left'] = to_terminal(left)
    else:
        data['left'] = get_split(left)
        split(data['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        data['right'] = to_terminal(right)
    else:
        data['right'] = get_split(right)
        split(data['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)

    return root


def predict(data, row):
    if row[data['index']] < data['value']:
        if isinstance(data['left'], dict):
            return predict(data['left'], row)
        else:
            return data['left']
    else:
        if isinstance(data['right'], dict):
            return predict(data['right'], row)
        else:
            return data['right']


def get_error(actual, predicted):
    incorrect = 0

    for i in range(len(actual)):
        if actual[i] != predicted[i]:
            incorrect += 1

    return incorrect / len(actual) * 100


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()

    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)

    return predictions


if __name__ == '__main__':
    train_data = read_data('hw7_train.dat')
    test_data = read_data('hw7_test.dat')

    predictions = decision_tree(train_data, train_data, 5, 1)
    Ein = get_error(train_data[:, -1], predictions)
    print(Ein)

    predictions = decision_tree(train_data, test_data, 5, 1)
    Eout = get_error(test_data[:, -1], predictions)
    print(Eout)
