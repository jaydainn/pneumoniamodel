import os
import cv2

import numpy as np


from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import pickle
from PIL import Image , ImageOps
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D , Reshape , Dropout
import tensorflow as tf
from datetime import datetime

# normal = 0 , pneumonia = 1 


def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

X_train = []
Y_train = []

X_test = []
Y_test = []

print(str(datetime.now()) + " starting data transformation")

for i in os.listdir("./data/train/NORMAL"):
    im = Image.open("./data/train/NORMAL/" + i  ) 
    imgrey = ImageOps.grayscale(im)
    imgrey.thumbnail((400,400))
    thumbnail = np.array(imgrey)
    imfinal = to_shape(thumbnail , (400, 400))
    X_train.append(imfinal)
    Y_train.append([1 , 0])
for i in os.listdir("./data/train/PNEUMONIA"):
    im = Image.open("./data/train/PNEUMONIA/" + i  ) 
    imgrey = ImageOps.grayscale(im)
    imgrey.thumbnail((400,400))
    thumbnail = np.array(imgrey)
    imfinal = to_shape(thumbnail , (400, 400))
    X_train.append(imfinal)
    Y_train.append([0 , 1])

for i in os.listdir("./data/test/NORMAL"):
    im = Image.open("./data/test/NORMAL/" + i  ) 
    imgrey = ImageOps.grayscale(im)
    imgrey.thumbnail((400,400))
    thumbnail = np.array(imgrey)
    imfinal = to_shape(thumbnail , (400, 400))
    X_test.append(imfinal)
    Y_test.append([1 , 0])
for i in os.listdir("./data/test/PNEUMONIA"):
    im = Image.open("./data/test/PNEUMONIA/" + i  ) 
    imgrey = ImageOps.grayscale(im)
    imgrey.thumbnail((400,400))
    thumbnail = np.array(imgrey)
    imfinal = to_shape(thumbnail , (400, 400))
    X_test.append(imfinal)
    Y_test.append([0 , 1])

Y_test = np.array(Y_test)
Y_train = np.array(Y_train)

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

X_train = tf.stack(X_train)
Y_train = tf.stack(Y_train)

print(str(datetime.now()) + " starting classifying ")
img_size = 400
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

n_classes = 2
n_channels = 1
filt_size = [5, 5] # 5x5 pixel filters

batch_size = 128
n_epochs = 30

model = Sequential()
model.add(Reshape([ img_size, img_size, 1]))
model.add(Conv2D(16, filt_size, padding='same',
                              activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2),
                                    padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(X_test,Y_test))

print(len(X_train))
print(len(Y_train))

print(str(datetime.now()) + " done")