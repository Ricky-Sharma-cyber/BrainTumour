import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical


image_directory='/content/drive/MyDrive/mri/Brain_tumor/dataset'
no_tumor_images= os.listdir(image_directory+ '/no')
yes_tumor_images= os.listdir(image_directory+ '/yes')
dataset=[]
label=[]
INPUT_SIZE=64


print(f"Image directory: {image_directory}")
for i , images_name in enumerate(no_tumor_images):
    if images_name.lower().endswith('.jpg'):

        image_path = os.path.join(image_directory, 'no', images_name)
        image=cv2.imread(image_path)
        if image is not None:
          print(f"Reading image: {image_path}")
          image=Image.fromarray(image, 'RGB')
          image=image.resize((INPUT_SIZE,INPUT_SIZE))
          dataset.append(np.array(image))
          label.append(0)

for i , images_name in enumerate(yes_tumor_images):
   if images_name.lower().endswith('.jpg'):

        image_path = os.path.join(image_directory, 'yes', images_name)
        image=cv2.imread(image_path)
        if image is not None:
          print(f"Reading image: {image_path}")
          image=Image.fromarray(image, 'RGB')
          image=image.resize((INPUT_SIZE,INPUT_SIZE))
          dataset.append(np.array(image))
          label.append(1)

dataset=np.array(dataset)
label=np.array(label)


x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

# x_train=normalize(np.array(x_train), axis=1)
# x_test=normalize(np.array(x_test), axis=1)

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)

# print(x_test.shape)
# print(y_test.shape)

#model building


model= Sequential()

model.add (Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add (Activation('relu'))
model.add (MaxPooling2D(pool_size=(2,2)))

model.add (Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add (Activation('relu')),
model.add (MaxPooling2D(pool_size=(2,2)))

model.add (Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add (Activation('relu'))
model.add (MaxPooling2D(pool_size=(2,2)))

model.add (Flatten())
model.add (Dense(64))
model.add (Activation('relu'))
model.add (Dropout(0.5))
model.add (Dense(2))
model.add (Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=16,
          verbose=1, epochs=10,
          validation_data=(x_test, y_test),
          shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')