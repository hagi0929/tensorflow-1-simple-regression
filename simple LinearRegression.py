import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

import numpy as np

print(tf.__version__)
x_data = np.array([[2, 4, 5], [4, 5, 6], [3, 6, 8], [3, 4, 6], [2, 4, 6], [3, 5, 6]])
y_data = np.array([3, 4, 5, 6, 7, 8])

model = Sequential()

model.add(Flatten(input_shape=(3,)))

model.add(Dense(1, activation='linear'))

model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')
model.summary()

hist = model.fit(x_data, y_data, epochs=1000)
print(model.predict(np.array([[2, 4, 5], [4, 5, 6], [3, 6, 8]])))