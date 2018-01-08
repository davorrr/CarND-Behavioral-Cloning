# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:49:01 2018

@author: davor
"""

import csv
import cv2
import numpy as np


lines = []

# Track 1 - driving forward 2x
with open('C:/Users/davor/Desktop/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) # Extracting path to the camera image for each line
        
# Track 1 - driving backward 2x
with open('C:/Users/davor/Desktop/data_unatrag/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) # Extracting path to the camera image for each line
        
# Track 2 - driving forward 2x
with open('C:/Users/davor/Desktop/data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) # Extracting path to the camera image for each line
        
# Track 2 - driving backward 2x
with open('C:/Users/davor/Desktop/data2_unatrag/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) # Extracting path to the camera image for each line
  
      
images = []
angles = []

for line in lines:
    for i in range(3):
        source_path = line[i]
        image_BGR = cv2.imread(source_path)
        image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        images.append(image_RGB)
        if i == 0:
            angle = float(line[3]) # Central camera steering angle
        elif i == 1:
            angle = float(line[3]) + 0.2 # Left camera steering angle
        elif i == 2:
            angle = float(line[3]) - 0.2 # Right camera steering angle
        angles.append(angle)
        
            
X_train = np.array(images)
y_train = np.array(angles)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


model = Sequential()

# Data preprocessing
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# NVIDIA model architecture implementation with dropout

model.add(Conv2D(24,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(36,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(48,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) 


# Training pipeline

TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, 
                         write_images=True)

model.compile(loss='mse', optimizer='adam') # Using mse intead of crossentropy
history_object = model.fit(X_train, y_train, validation_split = 0.2, epochs=3,
                           callbacks=[tbCallBack])


plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model mean sqaured error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')