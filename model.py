from keras import optimizers, backend
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Flatten, Dense, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd


def cropandresize(image):
    # Crop and resize image
    image = cv2.resize(image[50:140:], (192, 54))
    return image


def generator(samples, batch_size=32):
    num_samples = len(samples)

    # Steereeng angle correction
    corr = [0., .2, -.2]
    # Dividing batch size by 4 as used 3 images from left, center and right camera + one generated image
    batch_size = int(batch_size/4)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples-1, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:

                for i in range(0, 3):

                    path = batch_sample[i]
                    # filename = path.split('/')[-1]
                    path_sp = path.split('/')
                    current_path = './' + \
                        path_sp[-3] + '/' + path_sp[-2] + '/' + path_sp[-1]

                    if os.path.isfile(current_path):

                        image = cv2.imread(current_path)
                        # Convert image from BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Crop and resize image to 54x192
                        image = cropandresize(image)
                        images.append(image)

                        measurement = float(batch_sample[3])
                        measurements.append(measurement + corr[i])

                        # Generate flipped image and reverse steering angle
                        if i == 0:
                            image_flipped = np.fliplr(image)
                            images.append(image_flipped)
                            measurement_flipped = -measurement
                            measurements.append(measurement_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)


def load_samples(folder, file):
    lines = []

    with open('./' + folder + '/' + file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:

            if line[0].find(folder) > 0:
                add_flag = True
                for i in range(0, 3):

                    path = line[i]
                    current_path = './' + folder + \
                        '/IMG/' + path.split('/')[-1]

                    if not os.path.isfile(current_path):
                        add_flag = False
                        #print('Missing file. row not added')

                if add_flag:
                    lines.append(line)
    return lines


lines = []

# Load data from track 1
track1 = load_samples('track1', 'driving_log_updated.csv')
track1_len = len(track1)
print('Training samples from first track:', track1_len)


# Load data from track 2
track2 = load_samples('track2', 'driving_log.csv')
track2_len = len(track2)
print('Training samples from second track:', track2_len)

track1.extend(track2)
lines_len = len(track1)
print('Total training samples  :', lines_len)

lines = shuffle(track1)

# Split data to training and valitation set
train_samples, validation_samples = train_test_split(lines, test_size=0.1)

print('Training samples from 2 tracks:', len(train_samples),
      'Validation samples: ', len(validation_samples))


# Model architecture
# -----------------------------------------------------------
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(54, 192, 3)))
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(36, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[
              'mean_squared_error', 'mean_absolute_error'])
# ---------------------------------------------------------


batch_size = 32
epochs = 10

model.summary()


history_object = model.fit_generator(generator(train_samples), steps_per_epoch=len(train_samples*4)/batch_size, epochs=epochs,
                                     validation_data=generator(validation_samples), validation_steps=len(validation_samples*4)/batch_size, verbose=1)

model.save('model.h5')


# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
plt.savefig('mse.png')


exit()
