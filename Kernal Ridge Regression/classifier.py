import numpy as np


def read_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


def gaussian_rbf(gamma, x1, x2):
    delta = x1 - x2
    return np.exp(-gamma * np.dot(delta, delta.T))


def get_beta(gamma, lamb, X, Y):
    K = np.empty([X.shape[0], X.shape[0]])
    I = np.eye(X.shape[0])
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i, j] = gaussian_rbf(gamma, X[i], X[j])
            
    beta = np.linalg.inv(lamb * I + K).dot(Y)
    return beta


def predict(beta, gamma, x):
    delta = train_X - x
    kf = np.exp(-gamma * np.sum(delta ** 2, axis=1))
    score = beta.dot(kf)
    return np.sign(score)


def err(beta, gamma, X, Y):
    y_predict_list = []
    
    for i in range(Y.shape[0]):
        x = X[i,:]
        _predict = predict(beta, gamma, x)
        y_predict_list.append(_predict)
        
    return np.sum(Y != np.array(y_predict_list)) / len(Y)


if __name__ == '__main__':
    X, Y = read_data('hw2_lssvm_all.dat')
    train_X = X[:400, :]
    train_Y = Y[:400]
    test_X = X[400:, :]
    test_Y = Y[400:]
    
    for gamma in [32, 2, 0.125]:
        for lamb in [0.001, 1, 1000]:
            beta = get_beta(gamma, lamb, train_X, train_Y)
            Ein = err(beta, gamma, train_X, train_Y)
            Eout = err(beta, gamma, test_X, test_Y)
            print('gamma = {}, lambda = {}: Ein = {}, Eout = {}'.format(gamma, lamb, Ein, Eout))