import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from module.functions import read_csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
is_debug = True

# -------------------------------------------
# ファイル読み込み
# -------------------------------------------
(test_x, test_y, train_x, train_y) = read_csv(execute_path='./')

# -------------------------------------------
# モデル
# -------------------------------------------
input_num = 43
output_num = 43

model = Sequential()
model.add(Dense(5, activation='tanh', input_shape=(input_num,)))
model.add(Dense(output_num, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# -------------------------------------------
# 訓練
# -------------------------------------------
batch_size = 128
epochs = 30

history = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_x, test_y))

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ---------------------
# 保存
# ---------------------
model_file = 'files/time_series_model.h5'
model.save(model_file)
