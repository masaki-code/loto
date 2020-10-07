import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np

# ---------------------
# 訓練用データ
# ---------------------
# ファイル読み込み
train_path = '../file/train_mini.csv'
train = np.genfromtxt(train_path, delimiter=',')

# inputパラメータ：1パラメータ
# output：7パラメータ：本数字とボーナス数字で区別をつけた方が良いが一旦スルー
x_train = train[:, 0]
y_train = train[:, 2:]

# ---------------------
# テスト（検証）用データ
# ---------------------
test_path = '../file/test_mini.csv'
test = np.genfromtxt(test_path, delimiter=',')
x_test = train[:, 0]
y_test = train[:, 2:]

# ---------------------
# モデル
# ---------------------
# 総データ数：約1,500
# 訓練データ：1,000、テストデータ：500、くらいに分けてみる
# バーニーおじさんのルールにより、パラメータ数は100くらいで
input_num = 1
output_num = 7

# 以下で103パラメータなので、ちょうどいいくらい。
model = Sequential()
model.add(Dense(6, activation='relu', input_shape=(input_num,)))
model.add(Dropout(0.2))
model.add(Dense(6, activation='relu'))
model.add(Dropout(0.2))
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
model.save('loto_model_mini.h5')

# ---------------------
# 予測
# ---------------------
x = [1501]
result = model.predict(x, batch_size=None, verbose=0, steps=None)
print(result)
