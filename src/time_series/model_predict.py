import os
from keras.models import load_model
from module.functions import predict_nums, vector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_debug = True

# ---------------------
# 読み込み
# ---------------------
model_file = 'files/time_series_model.h5'
model = load_model(model_file)

if is_debug:
    model.summary()

# -------------------------------------------
# 予測
# -------------------------------------------
sample = [[1, 2, 3, 4, 5, 6]]
test = vector(sample)

result = model.predict(test, batch_size=None, verbose=0, steps=None)
result = result[0]

print('_________________________________________________________________')
print(test, '::', predict_nums(result))
print('_________________________________________________________________')
