import csv
import cv2
import numpy as np

# Load data from files
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Split training and validation data
from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(lines, test_size=0.2)

from sklearn.utils import shuffle

steering_correction = [0, 0.2, -0.2]
# Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3])
                    measurements.append(measurement + steering_correction[i])

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)

train_generator = generator(train_data)
validation_generator = generator(validation_data)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Model artchitecture
model = Sequential()
# Preprocessing - normalization and cropping
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
# Convolutions
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
# Flatten
model.add(Flatten())
# Full-connected
model.add(Dense(1164))
model.add(Dropout(0.2)) # Dropout to help prevent overfitting
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Plotting model architecture
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png', show_shapes=True)

# Training and validation
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_data)*6, validation_data=validation_generator, nb_val_samples=len(validation_data)*6, nb_epoch=5)

model.save('model.h5')
exit()
