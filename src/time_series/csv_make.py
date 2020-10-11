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
origin_path = '../../file/original/loto6.csv'
origin = np.genfromtxt(origin_path, skip_header=1, delimiter=',', dtype='int')

# origin = origin[:, 2:]
origin = origin[:, 2:8]  # ポーナス数字除く
_length = len(origin)

x = origin[0:_length - 1]
y = origin[1:_length]

x = vector(x)
y = vector(y)


# -------------------------------------------
# 分割
# -------------------------------------------
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


(test_x, test_y, train_x, train_y) = split(x, y)

is_debug = False
if is_debug:
    print(test_x, len(test_x), test_x[0])
    print(test_y, len(test_y), test_y[0])
    print(train_x, len(train_x), train_x[0])
    print(train_y, len(train_y), train_y[0])

np.savetxt('test_x_make.csv', test_x, delimiter=',', fmt='%d')
np.savetxt('test_y_make.csv', test_y, delimiter=',', fmt='%d')
np.savetxt('train_x_make.csv', train_x, delimiter=',', fmt='%d')
np.savetxt('train_y_make.csv', train_y, delimiter=',', fmt='%d')
