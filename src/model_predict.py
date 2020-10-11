from keras.models import load_model
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sample = 251

# is_debug = True
is_debug = False

# ---------------------
# 保存
# ---------------------
model_file = 'loto_model_big.h5'
model = load_model(model_file)

if is_debug:
    model.summary()

# ---------------------
# 予測
# ---------------------
test = int(sys.argv[1]) if len(sys.argv) > 1 else sample

test = [test]
result = model.predict(test, batch_size=None, verbose=0, steps=None)
result = result[0]

if is_debug:
    print('_________________________________________________________________')
    print('result', result)
    print('_________________________________________________________________')


def predict_nums(_x):
    k = 0
    _disc = {}
    for v in _x:
        k += 1
        _disc[k] = v

    _sorted = sorted(_disc.items(), key=lambda __x: __x[1], reverse=True)
    return sorted(list(map(lambda _x: _x[0], _sorted[::6])))


print('_________________________________________________________________')
print(test, '::', predict_nums(result))
print('_________________________________________________________________')
