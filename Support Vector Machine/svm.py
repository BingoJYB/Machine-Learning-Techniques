import math
import numpy as np
import pandas as pd

from sklearn import svm


def read_file(filename):
    return pd.read_csv(filename, delim_whitespace=True, names=['digit', 'intensity', 'symmetry'])


def divide_digit(dataframe, digit):
    dataframe['tag'] = np.where(dataframe['digit'] == digit, 1, -1)
    return dataframe


def calclute_error(predict_y, y):
    return (np.array(predict_y) != np.array(y)).sum()


def Q15(x, y):
    clf = svm.SVC(C=0.01, kernel='linear')
    clf.fit(x, y)
    print(np.linalg.norm(clf.coef_))


def Q16_Q17(x, y):
    clf = svm.SVC(C=0.01, kernel='poly', degree=2, gamma=1, coef0=1)
    clf.fit(x, y)
    dual_coefs = clf.dual_coef_[0]
    predict_y = clf.predict(x)
    return calclute_error(predict_y, y), dual_coefs


def Q18(x_train, y_train, x_test, y_test):
    sv_num = []
    Eout_list = []
    
    for c in [0.001, 0.01, 0.1, 1, 10]:
        clf = svm.SVC(C=c, kernel='rbf', gamma=100)
        clf.fit(x_train, y_train)
        
        predict_y = clf.predict(x_test)
        
        sv_num.append(len(clf.support_vectors_))
        Eout_list.append(calclute_error(predict_y, y_test))
        
    return sv_num, Eout_list


# Q15
dataframe = divide_digit(read_file('train.txt'), 0)
x = list(zip(dataframe['intensity'], dataframe['symmetry']))
y = dataframe['tag']
Q15(x, y)

# Q16&Q17
Ein_min = math.inf
dual_coef_max = -math.inf
for d in [0, 2, 4, 6, 8]:
    dataframe = divide_digit(read_file('train.txt'), d)
    x = list(zip(dataframe['intensity'], dataframe['symmetry']))
    y = dataframe['tag']
    Ein, dual_coefs = Q16_Q17(x, y)

    if Ein < Ein_min:
        Ein_min = Ein
        digit = d

    dual_coefs_sum = sum(abs(dual_coefs))
    if dual_coef_max < dual_coefs_sum:
        dual_coef_max = dual_coefs_sum

print(digit)
print(dual_coef_max)

# Q18
dataframe = divide_digit(read_file('train.txt'), 0)
x_train = list(zip(dataframe['intensity'], dataframe['symmetry']))
y_train = dataframe['tag']

dataframe = divide_digit(read_file('test.txt'), 0)
x_test = list(zip(dataframe['intensity'], dataframe['symmetry']))
y_test = dataframe['tag']

sv_num, Eout_list = Q18(x_train, y_train, x_test, y_test)

print(sv_num)
print(Eout_list)





