import pickle
import cv2 
import matplotlib.pyplot as plt
from PIL import Image, ImageOps 
import numpy as np
import os 
from keras.models import load_model
import tensorflow as tf 




def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

def predict_to_res(predictions):
    if(predictions == 0 ):
        return "Normal"
    else:
        return "Pneumonia"



# normal = 0 , pneumonia = 1 

with open('model-GradientBoostingClassifier-full.pkl', 'rb') as f:
    model = pickle.load(f)
j = 0 ; 
k = 0 ; 
subplot = 5
fig, axs = plt.subplots(2 , subplot)
for i in os.listdir("./data/val/PNEUMONIA"):
 im = Image.open("./data/val/PNEUMONIA/" + i  ) 
 imgrey = ImageOps.grayscale(im)
 imgrey.thumbnail((400,400))
 thumbnail = np.array(imgrey)
 imfinal = to_shape(thumbnail , (400, 400))
 imfinal = np.array(imfinal)
 predictions = model.predict([imfinal.flatten()])
 print(predictions[0])
 if(j < 5 ):
  axs[k][j].imshow(im , cmap='gray', vmin=0, vmax=255)
  axs[k][j].set_title(predict_to_res(predictions[0]))
 j = j+1

j = 0 
k = k + 1 

for i in os.listdir("./data/val/NORMAL"):
 im = Image.open("./data/val/NORMAL/" + i  ) 
 imgrey = ImageOps.grayscale(im)
 imgrey.thumbnail((400,400))
 thumbnail = np.array(imgrey)
 imfinal = to_shape(thumbnail , (400, 400))
 imfinal = np.array(imfinal)
 predictions = model.predict([imfinal.flatten()])
 print(predictions[0])
 if( j < 5 ):
  axs[k][j].imshow(im , cmap='gray', vmin=0, vmax=255)
  axs[k][j].set_title(predict_to_res(predictions[0]))
 j = j+1

axs[0]

plt.show()

  