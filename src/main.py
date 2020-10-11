import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.core import Activation

from keras.utils import np_utils

import numpy as np

import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ファイル読み込み
train_path = '../file/train_big.csv'
test_path = '../file/test_big.csv'
train = np.genfromtxt(train_path, delimiter=',', dtype='int')
test = np.genfromtxt(test_path, delimiter=',', dtype='int')

# ---------------------
# 関数
# ---------------------
input_num = 1
output_num = 43


def vector_y(_x, num=output_num):
    _list = np.empty((0, num), int)

    for __x in _x:
        n = np.add.reduce(np_utils.to_categorical(np.array(__x) - 1, num_classes=num))
        n = np.array([n])
        _list = np.append(_list, n, axis=0)

    return _list


# ---------------------
# 訓練用データ
# ---------------------
x_train = train[:, 0]
y_train = train[:, 2:]
y_train = vector_y(y_train)

# ---------------------
# テスト（検証）用データ
# ---------------------
x_test = test[:, 0]
y_test = test[:, 2:]
y_test = vector_y(y_test)

# ---------------------
# モデル
# ---------------------
# 総データ数：約1,500
# 訓練データ：1,000、テストデータ：500、くらいに分けてみる
# バーニーおじさんのルールにより、パラメータ数は100くらいで

# 以下で103パラメータなので、ちょうどいいくらい。
model = Sequential()
model.add(Dense(6, activation='tanh', input_shape=(input_num,)))
# model.add(Dense(6, activation='relu', input_shape=(input_num,)))
# model.add(Dropout(0.2))
# model.add(Dense(6, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(output_num, activation='softmax'))
model.add(Dense(output_num, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# ---------------------
# 訓練
# ---------------------
batch_size = 128
epochs = 20

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ---------------------
# 保存
# ---------------------
model.save('loto_model_big.h5')

# ---------------------
# 予測
# ---------------------
x = [251]
result = model.predict(x, batch_size=None, verbose=0, steps=None)
result = result[0]
print('==================')
print('result', result)
print('==================')


def trim(x):
    if x < 0.05:
        return 0
    else:
        return 1


trim_ret = list(map(trim, result))
print('==================')
print('trim', trim_ret)
print('==================')


def discs(x):
    k = 0
    _disc = {}
    for v in x:
        k += 1
        _disc[k] = v

    return _disc


discs_ret = discs(result)
print('==================')
print('discs', discs_ret)
print('==================')


def sort(x):
    return sorted(x.items(), key=lambda __x: __x[1], reverse=True)


sort_ret = sort(discs_ret)
print('==================')
print('sort', sort_ret)
print('==================')


y = sort_ret[::6]
print (y)

y = list(map(lambda x: x[0], y))
print (y)
