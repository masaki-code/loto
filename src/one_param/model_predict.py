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

    _result = sorted(_disc.items(), key=lambda __x: __x[1], reverse=True)
    _result = list(map(lambda _x: _x[0], _result))
    _result = _result[0:6]
    _result = sorted(_result)
    return _result


print('_________________________________________________________________')
print(test, '::', predict_nums(result))
print('_________________________________________________________________')

# for n in range(200):
#     result = model.predict([n], batch_size=None, verbose=0, steps=None)
#     print(n, '::', predict_nums(result[0]))
