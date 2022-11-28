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
    X_train.append(imfinal.flatten())
    Y_train.append(0)
for i in os.listdir("./data/train/PNEUMONIA"):
    im = Image.open("./data/train/PNEUMONIA/" + i  ) 
    imgrey = ImageOps.grayscale(im)
    imgrey.thumbnail((400,400))
    thumbnail = np.array(imgrey)
    imfinal = to_shape(thumbnail , (400, 400))
    X_train.append(imfinal.flatten())
    Y_train.append(1)

for i in os.listdir("./data/test/NORMAL"):
    im = Image.open("./data/test/NORMAL/" + i  ) 
    imgrey = ImageOps.grayscale(im)
    imgrey.thumbnail((400,400))
    thumbnail = np.array(imgrey)
    imfinal = to_shape(thumbnail , (400, 400))
    X_train.append(imfinal.flatten())
    Y_train.append(0)
for i in os.listdir("./data/test/PNEUMONIA"):
    im = Image.open("./data/test/PNEUMONIA/" + i  ) 
    imgrey = ImageOps.grayscale(im)
    imgrey.thumbnail((400,400))
    thumbnail = np.array(imgrey)
    imfinal = to_shape(thumbnail , (400, 400))
    X_train.append(imfinal.flatten())
    Y_train.append(1)

Y_test = np.array(Y_test)
Y_train = np.array(Y_train)
print(str(datetime.now()) + " starting classifying ")
grad = GradientBoostingClassifier(verbose=1)
grad.fit( X_train , Y_train.flatten() )
Y_pred = grad.predict(X_test)
accuracy = 1 - mean_squared_error(Y_test , Y_pred)
print("accuracy: "  +  str(accuracy))
print("erreur: " + str((1 - accuracy)))
# now you can save it to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(grad, f)

print(str(datetime.now()) + " done")
