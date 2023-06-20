import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

training_images= ('/Users/haqueanamul/PycharmProjects/Driver drwsiness/Pneumonia/train/')
validation_images=('/Users/haqueanamul/PycharmProjects/Driver drwsiness/Pneumonia/val/')
testing_images=('/Users/haqueanamul/PycharmProjects/Driver drwsiness/Pneumonia/test/')

normal_cases_images= ('/Users/haqueanamul/PycharmProjects/Driver drwsiness/Pneumonia/train/NORMAL')

pneumonia_cases_images= ('training_images/PNEUMONIA')
IMAGE_SIZE=(150,150)
IMAGE_CHANNELS=3
batch_size=42

train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

valid_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)


train_generator = train_data.flow_from_directory(training_images, target_size=IMAGE_SIZE, batch_size=batch_size,
                                                    class_mode="binary")

validation_generator = valid_data.flow_from_directory(validation_images, target_size=IMAGE_SIZE,
                                                              batch_size=batch_size, class_mode="binary")

test_generator = test_data.flow_from_directory(testing_images, target_size=IMAGE_SIZE, batch_size=batch_size,
                                                  class_mode="binary")


model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),


    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),


    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.summary()
class EarlyStopping(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.96):
      print("\nDesired accuracy reached. Stopping training...")
      self.model.stop_training = True
early_stopping = EarlyStopping()

model.compile(loss= "binary_crossentropy", optimizer=RMSprop(learning_rate=0.001), metrics=["accuracy"])
history = model.fit(train_generator,epochs=20,validation_data=validation_generator,
                    callbacks=[early_stopping])

test_accuracy = model.evaluate(test_generator)
print('The accuracy on test set :',test_accuracy[1]*100 )