from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import os
import csv
import cv2
import numpy as np
import sklearn
from math import ceil
import pandas as pd

number_of_epochs = 8
learning_rate = 1e-4
activation_relu = 'relu'


model = Sequential()

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), input_shape=(80, 160, 3)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation(activation_relu))
model.add(Dropout(0.25))


model.add(Dense(50))
model.add(Activation(activation_relu))
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))


model.compile(optimizer=Adam(learning_rate), loss="mse")

samples = pd.read_csv("../data/driving_log.processed.3.csv")

print(samples.head(10))

batch_size=32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    samples,
    directory = "",
    x_col="image",
    y_col="steering",
    target_size=(80, 160),
    batch_size=batch_size,
    subset="training",
    class_mode="other"
)

val_gen = datagen.flow_from_dataframe(
    samples,
    directory = "",
    x_col="image",
    y_col="steering",
    target_size=(80, 160),
    batch_size=batch_size,
    subset="validation",
    class_mode="other"
)


hist = model.fit_generator(
    train_gen,
    epochs=number_of_epochs,
    validation_data=val_gen
)

model.save('model.h5')




