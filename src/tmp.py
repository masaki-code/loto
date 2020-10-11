import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import backend as K

def tanh(x):
    return K.tanh(x)

def my_activation(x):
    if x < 0:
        return 0
    elif x <= 43:
        return x
    else:
        return 0

from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


model = Sequential()
model.add(Dense(6, activation=my_activation, input_shape=(1,)))
# model.add(Dense(1, activation=my_activation))
# model.add(Activation(my_activation))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


print(my_activation(10))
print(my_activation(-1))
print(my_activation(43))
print(my_activation(45))
