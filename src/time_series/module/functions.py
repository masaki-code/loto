import numpy as np
from keras.utils import np_utils

input_num = 43
output_num = 43


def vector(_x, num=output_num):
    _list = np.empty((0, num), int)

    for __x in _x:
        n = np.add.reduce(np_utils.to_categorical(np.array(__x) - 1, num_classes=num))
        n = np.array([n])
        _list = np.append(_list, n, axis=0)

    return _list


def predict_nums(_x):
    k = 0
    _disc = {}
    for v in _x:
        k += 1
        _disc[k] = v

    _result = sorted(_disc.items(), key=lambda __x: __x[1], reverse=True)
    _result = list(map(lambda _x: _x[0], _result))
    _result = _result[0:6]
    _result = sorted(_result)
    return _result


def split(x, y):
    num_all = len(y)
    num_train = 1250
    num_test = num_all - num_train

    id_all = np.random.choice(num_all, num_all, replace=False)
    id_test = id_all[0:num_test]
    id_train = id_all[num_test:num_all]

    _test_x = x[id_test]
    _test_y = y[id_test]
    _train_x = x[id_train]
    _train_y = y[id_train]

    return _test_x, _test_y, _train_x, _train_y


def write_csv(test_x, test_y, train_x, train_y):
    np.savetxt('files/test_x_make.csv', test_x, delimiter=',', fmt='%d')
    np.savetxt('files/test_y_make.csv', test_y, delimiter=',', fmt='%d')
    np.savetxt('files/train_x_make.csv', train_x, delimiter=',', fmt='%d')
    np.savetxt('files/train_y_make.csv', train_y, delimiter=',', fmt='%d')


def read_csv():
    _test_x = np.genfromtxt('../files/test_x_make.csv', skip_header=1, delimiter=',', dtype='int')
    _test_y = np.genfromtxt('../files/test_y_make.csv', skip_header=1, delimiter=',', dtype='int')
    _train_x = np.genfromtxt('../files/train_x_make.csv', skip_header=1, delimiter=',', dtype='int')
    _train_y = np.genfromtxt('../files/train_y_make.csv', skip_header=1, delimiter=',', dtype='int')

    return _test_x, _test_y, _train_x, _train_y
