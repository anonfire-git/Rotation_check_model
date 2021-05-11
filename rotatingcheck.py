import pymongo
from pymongo import MongoClient
from matplotlib import pyplot
from scipy import signal
import numpy as np
import pylab as pl
import time
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

# Connection to DB
client = MongoClient('mongodb://172.16.98.148:27017/')
db = client.microphone_measurements
collection = db.measurements

all_measurements = collection.find({})

# Building the training data arrays
rotating_array = []
A_array = []

for single_measurement in all_measurements:
    rotating = single_measurement['rotating']
    data = single_measurement['data']
    data_numpy = np.array(data)
    #data_copy = np.copy(data_numpy)
    #data_nor = ( data_numpy / 2**24 )
    #data_nor = ( (data_numpy - data_numpy.min()) / (data_numpy.max() - data_numpy.min()))
    _, _, A = signal.spectrogram(data_numpy)

    A_array.append(np.array(A))

    # 2 node
    # if rotating:
    #     rotating_array.append(np.array([1,0]))
    # else:
    #     rotating_array.append(np.array([0,1]))

    if rotating:
        rotating_array.append(np.array([1]))
    else:
        rotating_array.append(np.array([0]))


rotating_array_numpy = np.stack(rotating_array)


A_array_numpy = np.stack(A_array)



A_min = 8.217198470548396
A_max = 6033455016667724.0

# normalization of amplitude data
A_nor = (A_array_numpy-A_array_numpy.min()) / (A_array_numpy.max()-A_array_numpy.min())




A_nor = A_nor.reshape( A_nor.shape + (1,) )

print (A_nor.shape)
print(rotating_array_numpy.shape)


x_train, x_test, y_train, y_test = train_test_split(
    A_nor,
    rotating_array_numpy,
    test_size=0.33,
)



# Here I build the neural network
model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train[0].shape, activation="elu"))
model.add(Conv2D(16, (3, 3), activation="elu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (3, 3), padding='same', activation="elu"))
model.add(Conv2D(32, (3, 3), activation="elu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))



model.add(Flatten())
model.add(Dense(units=216, activation="elu"))
model.add(Dropout(0.2))
model.add(Dense(units=y_train.shape[1], activation="sigmoid"))
# model.add(Dense(units=rotating_array_numpy.shape[1], activation="softmax"))

print(model.summary())
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=10)

'''
for i in range(len(A_nor)):
    print(A_nor.shape)
    y = model.predict(  np.expand_dims(A_nor[i] , axis=0)  )
    if y < 0.5:
        y = 0
    elif y >= 0.5:
        y = 1
    t = rotating_array_numpy[i]
    print(y,t)
'''

evaluation_result = model.evaluate(x_test, y_test) 

print(evaluation_result)

model.save('./rotating-check/1/') 