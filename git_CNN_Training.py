
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import random
import os
import pickle
import time

# Building the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
#from keras_svm import SVM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy.random import seed
seed(42)
random.set_seed(42)


early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(r'/home/file.h5', verbose=1, save_best_only=True)

# Build CNN
# Initialising the CNN
classifier = Sequential()
# 1st Convolutional and Pooling layers

classifier.add(Convolution2D(40, (5, 5),  padding="same",
                             input_shape=(32, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding Second convolutional and Pooling layers
classifier.add(Convolution2D(30, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding Third Convolutional and Pooling Layer
classifier.add(Convolution2D(50, (2, 2), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening
classifier.add(Flatten())
classifier.add(Dense(units=300, activation='relu'))
classifier.add(Dense(units=2, activation='softmax'))
classifier.summary()

# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting CNN to images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

path = r"/home/images"
os.chdir(path)
# training_set = train_datagen.flow_from_directory('Train',
training_set = train_datagen.flow_from_directory('Train',
                                                 target_size=(32, 128),
                                                 batch_size=42,  # 84,##42,
                                                 class_mode='categorical', shuffle=True, seed=42)
# test_set = test_datagen.flow_from_directory('Test',
test_set = test_datagen.flow_from_directory('Test',
                                            target_size=(32, 128),
                                            batch_size=35,
                                            class_mode='categorical', shuffle=True, seed=42)
startTime = time.time()
os.chdir(path)
hist = classifier.fit_generator(training_set,
                                steps_per_epoch=354,  # 350,
                                epochs=100,
                                callbacks=[early_stopping, model_checkpoint],
                                validation_data=test_set,
                                validation_steps=100,  # 262,
                                workers=8)

endTime = time.time()
print("Time taken : ", endTime - startTime)

with open('/home/filename', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
