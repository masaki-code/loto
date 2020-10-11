import os
import numpy as np
from keras.utils import np_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

output_num = 43

is_debug = True


# -------------------------------------------
# 関数定義
# -------------------------------------------
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


# -------------------------------------------
# ファイル読み込み
# -------------------------------------------
def read():
    _test_x = np.genfromtxt('test_x_make.csv', skip_header=1, delimiter=',', dtype='int')
    _test_y = np.genfromtxt('test_y_make.csv', skip_header=1, delimiter=',', dtype='int')
    _train_x = np.genfromtxt('train_x_make.csv', skip_header=1, delimiter=',', dtype='int')
    _train_y = np.genfromtxt('train_y_make.csv', skip_header=1, delimiter=',', dtype='int')

    return _test_x, _test_y, _train_x, _train_y


(test_x, test_y, train_x, train_y) = read()


# -------------------------------------------
# ファイル読み込み
# -------------------------------------------
print(test_x, test_y, train_x, train_y)