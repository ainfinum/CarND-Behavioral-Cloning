# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/track1.jpg "Track 1"
[image2]: ./images/track2.jpg "Track 2"
[image3]: ./images/recovery.jpg "Recovery images"
[image4]: ./images/counted_samples.png "Samples bar chart"
[image5]: ./images/counted_bal.png "Balanced samples bar chart"
[image6]: ./images/preprocessed.jpg "Cropped and resized images"
[image7]: ./images/flipped.jpg "Flipped Image"
[image8]: ./images/conv-netw.png  "Model architecture"

[video1]: ./track1.mp4 "Video track 1"


## 1. Project files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md a report summarizing the results
* track1.mp4, track2.mp4 - output videos for track #1 and #2 


## 2. Dataset
The provided driving simulator has two different tracks. The first track is the curving road on a flat surface, but the second track is more challenging with very sharp bends that can throw the car off the track. The simulator outputs steering angle and corresponding images 160x320x3 pixels from 3 different camera viewpoints - center, left and right.
 
### 2.1 Data collection
Training data for the project have been obtained by manually driving the car in the simulator around the tracks. It contains data collected driving in the middle of the road for several laps 
and in "recovery" mode - moving the car to the side of the road and drive it back to the center of the road. Also, I collected the data driving counter-clockwise and I recorded additionally the hardest turns of the tracks.  

Left, center and right cameras
![alt text][image1]
![alt text][image2]

Recovery mode
![alt text][image3]

As can be seen from the bar chart below the collected data was unbalanced with the spike near the center, so I decided to remove some data to balance the training set (remove.py line 129).

![alt text][image4]

Balanced training set

![alt text][image5]

### 2.2 Data augmentation

After collection and balancing the data I ended up with the following set:

Training samples from the first track: 28506

Training samples from the second track: 30194

Total training samples from 2 tracks: 58700

Training samples: 52830 

Validation samples:  5870


The training data can be downloaded from the google drive:

[Track 1](https://drive.google.com/open?id=1ygSNh7AXM2q21QsOD7qSfio9EtIbcuC9)

[Track 2](https://drive.google.com/open?id=1eEzf_kWU59aoMeh94nVc02Ar5KncF2iq)


In order to add more training data I used the following methods:
* Left and right cameras. Each sample received from the simulator consist of 3 images from different camera positions: left, center and right. So I used left and right cameras images after applying steering angle correction of 0.2. This will in crease amount of training data by a factor of 3

* Horizontal flip. I flipped every center image and changed the sign of the steering angle

![alt text][image7]

### 2.3 Data preprocessing

* Crop images by 50 pixels from the top and 20 pixels from the bottom of the image and resize to a shape of 192 by 54.

![alt text][image6]



## 3. Model Architecture and Training Strategy

### 3.1. Solution Design Approach

My first idea was to use InceptionV3 network with pretrained weights. I removed the top and bottom layers and trained it for several epochs. I didn't get a good result from the first try and considering that the weights of InceptionV3 is a huge file I decided to search for other model architecture that can be used for this task. I found [Nvidia](https://github.com/user/repo/blob/branch/other_file.md) paper and created similar simplified model architecture with 4 convolutional layers and 3 fully connected layers with dropout on 2 out of 3 dense layers to prevent overfitting. 
 
I split my data set into a training and validation set. I found that the model had a similar mean squared error on the training set on the validation set. It shows me that my model working fine so I trained it for 10 epochs.

The next step was to run the simulator to test the model driving around track one. The car fell off the track in one spot so I collected more data to improve the driving behavior.
After training the model for 15 epochs on the data collected from 2 track the model were able to drive autonomously around both tracks without leaving the road.

### 3.2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the 4 convolutional layers and 3 fully connected layers (model.py line 129).

Here is a visualization of the architecture:

![alt text][image8]


 
|Layer (type)                 |Output Shape            |Param#|   
|----------------------------:|:----------------------:|:----:|
|lambda_1 (Lambda)            |(None, 54, 192, 3)      |0     |      
|conv2d_1 (Conv2D)            |(None, 52, 190, 24)     |672   |    
|max_pooling2d_1 (MaxPooling2)|(None, 26, 95, 24)      |0     |     
|conv2d_2 (Conv2D)            |(None, 24, 93, 36)      |7812  |     
|max_pooling2d_2 (MaxPooling2)|(None, 12, 46, 36)      |0     |     
|conv2d_3 (Conv2D)            |(None, 10, 44, 48)      |15600 |    
|max_pooling2d_3 (MaxPooling2)|(None, 5, 22, 48)       |0     |    
|conv2d_4 (Conv2D)            |(None, 3, 20, 64)       |27712 |    
|max_pooling2d_4 (MaxPooling2)|(None, 1, 10, 64)       |0     |    
|flatten_1 (Flatten)          |(None, 640)             |0     |    
|dense_1 (Dense)              |(None, 256)             |164096|    
|dropout_1 (Dropout)          |(None, 256)             |0     |    
|dense_2 (Dense)              |(None, 50)              |12850 |     
|dropout_2 (Dropout)          |(None, 50)              |0     |      
|dense_3 (Dense)              |(None, 10)              |510   |       
|dense_4 (Dense)              |(None, 1)               |11    |
|                             |                        |      |      
|Total params:                |229,263                 |      |
|Trainable params:            |229,263                 |      |
|Non-trainable params:        |0                       |      |




### 3.3 Training Process

I finally randomly shuffled the 58700 training samples from 2 tracks and put 10% of the data into a validation set. 

52830 training samples were used for training the model. I trained the model for 15 epochs to get low mean squared error. The adam optimizer was used so that manually training the learning rate wasn't necessary.

Considering that I collected a good amount of training data and I will have additional images from the left and right cameras and flipped image it will use a lot of GPU/CPU memory. In order to avoid issues with computer memory, I created a generator() (model.py line 24) that generate batches for Keras fit_generator() function.  


6764/6763 [==============================] - 239s - loss: 0.0491 - mean_squared_error: 0.0491 - mean_absolute_error: 0.1668 - val_loss: 0.0380 - val_mean_squared_error: 0.0380 - val_mean_absolute_error: 0.1456
Epoch 2/15
6764/6763 [==============================] - 234s - loss: 0.0370 - mean_squared_error: 0.0370 - mean_absolute_error: 0.1444 - val_loss: 0.0330 - val_mean_squared_error: 0.0330 - val_mean_absolute_error: 0.1352
Epoch 3/15
6764/6763 [==============================] - 233s - loss: 0.0324 - mean_squared_error: 0.0324 - mean_absolute_error: 0.1348 - val_loss: 0.0311 - val_mean_squared_error: 0.0311 - val_mean_absolute_error: 0.1309
Epoch 4/15
6764/6763 [==============================] - 233s - loss: 0.0297 - mean_squared_error: 0.0297 - mean_absolute_error: 0.1290 - val_loss: 0.0287 - val_mean_squared_error: 0.0287 - val_mean_absolute_error: 0.1262
Epoch 5/15
6764/6763 [==============================] - 234s - loss: 0.0278 - mean_squared_error: 0.0278 - mean_absolute_error: 0.1248 - val_loss: 0.0270 - val_mean_squared_error: 0.0270 - val_mean_absolute_error: 0.1212
Epoch 6/15
6764/6763 [==============================] - 236s - loss: 0.0267 - mean_squared_error: 0.0267 - mean_absolute_error: 0.1222 - val_loss: 0.0263 - val_mean_squared_error: 0.0263 - val_mean_absolute_error: 0.1195
Epoch 7/15
6764/6763 [==============================] - 237s - loss: 0.0256 - mean_squared_error: 0.0256 - mean_absolute_error: 0.1195 - val_loss: 0.0238 - val_mean_squared_error: 0.0238 - val_mean_absolute_error: 0.1139
Epoch 8/15
6764/6763 [==============================] - 236s - loss: 0.0247 - mean_squared_error: 0.0247 - mean_absolute_error: 0.1172 - mval_loss: 0.0237 - val_mean_squared_error: 0.0237 - val_mean_absolute_error: 0.1134
Epoch 9/15
6764/6763 [==============================] - 235s - loss: 0.0239 - mean_squared_error: 0.0239 - mean_absolute_error: 0.1153 - val_loss: 0.0247 - val_mean_squared_error: 0.0247 - val_mean_absolute_error: 0.1156
Epoch 10/15
6764/6763 [==============================] - 233s - loss: 0.0233 - mean_squared_error: 0.0233 - mean_absolute_error: 0.1140 - val_loss: 0.0234 - val_mean_squared_error: 0.0234 - val_mean_absolute_error: 0.1127
Epoch 11/15
6764/6763 [==============================] - 235s - loss: 0.0226 - mean_squared_error: 0.0226 - mean_absolute_error: 0.1123 - val_loss: 0.0247 - val_mean_squared_error: 0.0247 - val_mean_absolute_error: 0.1159
Epoch 12/15
6764/6763 [==============================] - 234s - loss: 0.0222 - mean_squared_error: 0.0222 - mean_absolute_error: 0.1112 - val_loss: 0.0218 - val_mean_squared_error: 0.0218 - val_mean_absolute_error: 0.1081
Epoch 13/15
6764/6763 [==============================] - 235s - loss: 0.0218 - mean_squared_error: 0.0218 - mean_absolute_error: 0.1101 - val_loss: 0.0228 - val_mean_squared_error: 0.0228 - val_mean_absolute_error: 0.1111
Epoch 14/15
6764/6763 [==============================] - 235s - loss: 0.0215 - mean_squared_error: 0.0215 - mean_absolute_error: 0.1093 - val_loss: 0.0216 - val_mean_squared_error: 0.0216 - val_mean_absolute_error: 0.1075
Epoch 15/15
6764/6763 [==============================] - 235s - loss: 0.0210 - mean_squared_error: 0.0210 - mean_absolute_error: 0.1082 - val_loss: 0.0214 - val_mean_squared_error: 0.0214 - val_mean_absolute_error: 0.1073

 


## 4. Result

The model was tested by running it through the simulator and the car can be driven autonomously around both tracks.
I think the result of the CNN model is pretty impressive. The model was trained less than 60 minutes and it can drive a car on a quite challenging road.

Video from track 1

[![Track 1 youtube video](https://img.youtube.com/vi/nmmvm4ZCuO0/default.jpg)](https://youtu.be/nmmvm4ZCuO0)

[Track 1 video, github](./track1.mp4)

Video from track 2

[![Track 1 youtube video](https://img.youtube.com/vi/TGKVMtZpG2Y/default.jpg)](https://youtu.be/TGKVMtZpG2Y)

[Track 2 video, github](./track1.mp4)

This result gives a rough idea of what this model is capable of. I've tried to run the model on the first track with the speed of 30 MPH but the car fell off the road.
Perhaps for high speed the model should be trained more on data collected from the track 1.

It would be nice to test it in the real world so I'm planning to use this model to control RC car with a camera and Raspberry Pi 3 B+ on board.
