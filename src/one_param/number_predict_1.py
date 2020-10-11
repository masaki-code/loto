import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers.core import Activation

import numpy as np

import tensorflow as tf

# ファイル読み込み
train_path = '../../file/train_big.csv'
test_path = '../../file/test_big.csv'

# ---------------------
# 訓練用データ
# ---------------------
train = np.genfromtxt(train_path, delimiter=',', dtype='int')
x_train = train[:, 0]
y_train = train[:, 2:]

# ---------------------
# テスト（検証）用データ
# ---------------------
test = np.genfromtxt(test_path, delimiter=',', dtype='int')
x_test = test[:, 0]
y_test = test[:, 2:]

# ---------------------
# モデル
# ---------------------
# 総データ数：約1,500
# 訓練データ：1,000、テストデータ：500、くらいに分けてみる
# バーニーおじさんのルールにより、パラメータ数は100くらいで
input_num = 1
output_num = 7


# 以下で103パラメータなので、ちょうどいいくらい。
# model = Sequential()
# model.add(Dense(6, activation='relu', input_shape=(input_num,)))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(output_num, activation='softmax'))
# model.summary()

# outputの活性化関数をreluにしたら多少改善
# model = Sequential()
# model.add(Dense(6, activation='relu', input_shape=(input_num,)))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(output_num, activation='relu'))
# model.summary()

# 自作の活性化関数にしたい。。

def my_activation(_x):
    x = _x
    x = tf.where(x < 0, 0.0, x)
    x = tf.where(x > 99999, 0.0, x)
    return x


model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(input_num,)))
model.add(Dense(output_num, activation=my_activation))
model.summary()

# model = Sequential()
# model.add(Dense(6, activation='relu', input_shape=(input_num,)))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(output_num, activation=my_activation_2))
# # model.add(Dense(output_num, activation=my_activation))
# model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------
# 訓練
# ---------------------
batch_size = 128
epochs = 5

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
x = [1501]
result = model.predict(x, batch_size=None, verbose=0, steps=None)
result = result[0]
print(result)
