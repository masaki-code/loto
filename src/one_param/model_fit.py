import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import np_utils

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ファイル読み込み
train_path = '../../file/train_big.csv'
test_path = '../../file/test_big.csv'
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
# バーニーおじさんのルールにより、パラメータ数は100くらいが理想？

# 結果の次元が43あるのでパラメータ数を適切にするには難しそう。モデルを見直した方が良い？
model = Sequential()
model.add(Dense(6, activation='tanh', input_shape=(input_num,)))
# model.add(Dense(6, activation='relu', input_shape=(input_num,)))
# model.add(Dense(10, activation='selu', input_shape=(input_num,)))
# model.add(Dense(10, activation='selu'))
model.add(Dense(output_num, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# ---------------------
# 訓練
# ---------------------
batch_size = 128
epochs = 100

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
model_file = 'loto_model_big.h5'
model.save(model_file)
