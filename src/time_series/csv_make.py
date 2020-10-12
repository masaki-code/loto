import os
import numpy as np
from module.functions import vector, split, write_csv, write_csv_origin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
output_num = 43
is_debug = True

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

# -------------------------------------------
# 分割
# -------------------------------------------
(test_x, test_y, train_x, train_y) = split(x, y)

if is_debug:
    print(test_x, len(test_x), test_x[0])
    print(test_y, len(test_y), test_y[0])
    print(train_x, len(train_x), train_x[0])
    print(train_y, len(train_y), train_y[0])

write_csv_origin(test_x, test_y, train_x, train_y)

test_x = vector(test_x)
test_y = vector(test_y)
train_x = vector(train_x)
train_y = vector(train_y)

write_csv(test_x, test_y, train_x, train_y)
