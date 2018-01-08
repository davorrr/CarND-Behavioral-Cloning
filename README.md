# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_12_27_16_52_32_745.jpg "First track example"
[image2]: ./examples/center_2018_01_02_22_44_25_006.jpg "Second track example"
[image3]: ./examples/Camera_positon.PNG "Camera positions"
[image4]: ./examples/left_2017_12_27_16_52_30_132.jpg "Left camera image"
[image5]: ./examples/center_2017_12_27_16_52_30_132.jpg "Center camera image"
[image6]: ./examples/right_2017_12_27_16_52_30_132.jpg "Right camera image"
[image7]: ./examples/Network_architecture.PNG "Network architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
# Behavioural Cloning Project

## The Project consits of the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The entire code used to train and save the convolutional neural network is in the model.py file. In the file there is a pipeline used for training and validating the model.

## Model Architecture and Training Strategy

### 1. Model architecture

In total 3 models were used as a starting point for deriving the model that would be capable of driving the car in the simulator: 
* custom convolutional neural network
* neural network used by NVIDIA's team in [End-to-End Learning for Self-Driving, Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper,
* neural network used by Comma.ai team in the [Learning a Driving Simulator](https://arxiv.org/pdf/1608.01230.pdf) paper.

For the purpouse of this implementation on the collected dataset the model employed NVIDIA's team showed best performance by succesfully finishing the both tracks in accordance with the project specifications (No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe).

Prior to introducing data into the model the dataset was normalised using a Keras Lambda layer:
```sh
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
```
Afterwards the images were cropped to remove the potrions that do no carry relevant infomation - bottom part of the photo which was mostly covered by vehicle's hood and the top part which mostly shows the sky.
```sh
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

The code for the model pipeline is:
```sh
model.add(Conv2D(24,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(36,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(48,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(60,(5,5), activation='relu', strides=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```
As can be seen the code is written using a Keras wrapper for TensorFlow and it uses 7 Convolutional layers with ReLU activation functions to introduce non-linearity into the model. In the first 4 convolutional layers 2x2 strides were used while in the following 3 convolutional layers default 1x1 stride configuration was kept. After the Convolutional section we flatten the model and then use 4 fully connected layers to produce the steering values.

### 2. Overfitting reduction
Testing was done to see if incorporating Dropout would lead to even better results but that was not the case, at least with the amount of the training data used. The strategy to reduce overfitting relied on reducing the number of training epochs. During training validation loss was monotonically decreasing usually for up to 3 epochs so that was the number of epochs chosen.

### 3. Model parameter training
The model used an adam optimizer, so the learning rate was not tuned manually.
```sh
model.compile(loss='mse', optimizer='adam')
```

### 4. Training data 
Training data was chosen to keep the vehicle driving on the road. For training data images from both tracks were used to insure that the model was able to generalize better. 

For details about how I created the training data, see the next section. 


## Solution Design Approach

### 1. Data collection

My first step was to collect an apropriate dataset which would capture desired driving behavior that then could be used in training the final model. In order to do that I used both tracks where I collected images from 2 laps of counterclockwise and 2 laps of clockwise driving. Here are the examples of driving from both tracks:

![alt text][image1]
![alt text][image2]

Data from both tracks were incorporated into the dataset so that the model would generalise better and I also had an ambition to build the model that would perform equally good on both tracks. The data from all three cameras were used with added correction for steering angles. The correction was necessary because of the position of the cameras on the car

![alt text][image3]

The two side cameras allow the model to better learn to follow the edges of the track and also to be able to recover if the car approaches the edge.

![alt text][image4]
![alt text][image5]
![alt text][image6]

The entire collection process gave 24873 data points which were then randomly shuffled and then split 4:1 into training and validation set so that the performance of the model could be gauged and also to determine if the model was overfitting.


### 2. Model synthesis


The overall strategy for deriving a model was first to attempt to build a simple neural network in order to get a feel for the problem and then to use an existing proven architecture which would be adapted to the problem.

As was already mentioned 2 existing architectures were used NVIDIA's and Comma.ai's. Both were trained from the start and both could drive the car around the first track. However Comma.ai's model could not be trained to safely drive around the second track so on the end NVIDIA's model was chosen as a final solution.

![alt text][image7]

Attempts to improve the NVIDIA's model by adding dropout were made but for the dataset used they actually degraded the model. I assume that dropout would perform better on a larger dataset (NVIDIA's team used 72 hours of driving to train the model). Also attmepts to reduce model size were made but then the model was not able to learn small differences between the piece of track the car is currently driving on and the adjoining track wich lead to the car driving of the road. Also attempt was made to remove the datapoints with 0° driving angles to remove the bias for straight line driving and improve cornering but then the car would steer of the track on straights.

The ideal number of epochs was chosen to be 3. On a number of epochs higher than 3 validation loss would start to monotonically increas which would indicate overfitting. In the process of training Adam optimizer was used.

On the end the vechicle was capable to autonomously drive around both tracks.






