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
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
import pickle

from datetime import datetime

# normal = 0 , pneumonia = 1 


X_train = []
Y_train = []

X_test = []
Y_test = []

print(str(datetime.now()) + " starting data transformation")

for i in os.listdir("./data/train/NORMAL"):
    im = cv2.imread("./data/train/NORMAL/" + i , cv2.IMREAD_GRAYSCALE) 
    im.resize((400 , 400))
    X_train.append(im.flatten())
    Y_train.append(0)
for i in os.listdir("./data/train/PNEUMONIA"):
    im = cv2.imread("./data/train/PNEUMONIA/" + i , cv2.IMREAD_GRAYSCALE)
    im.resize((400 , 400))
    X_train.append(im.flatten())
    Y_train.append(1)

for i in os.listdir("./data/test/NORMAL"):
    im = cv2.imread("./data/test/NORMAL/" + i , cv2.IMREAD_GRAYSCALE)
    im.resize((400 , 400))
    X_test.append(im.flatten())
    Y_test.append(0)
for i in os.listdir("./data/test/PNEUMONIA"):
    im = cv2.imread("./data/test/PNEUMONIA/" + i , cv2.IMREAD_GRAYSCALE)
    im.resize((400 , 400))
    X_test.append(im.flatten())
    Y_test.append(1)


Y_test = np.array(Y_test)
Y_train = np.array(Y_train)

print(str(datetime.now()) + " starting classifying ")

grad = GradientBoostingClassifier()
score = cross_validate(grad , X_train , Y_train.flatten() )
accuracy = score['test_score'].mean()
print("accuracy: "  +  str(accuracy))
print("erreur: " + str((1 - accuracy)))
# now you can save it to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(grad, f)

print(str(datetime.now()) + " done")