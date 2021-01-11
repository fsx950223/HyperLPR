#coding=utf-8
import tensorflow as tf

import cv2
import numpy as np


plateType  = ["蓝牌","单层黄牌","新能源车牌","白色","黑色-港澳"]
def Getmodel_tensorflow(nb_classes):
    # nb_classes = len(charset)

    img_rows, img_cols = 9, 34
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (5, 5),input_shape=(img_rows, img_cols,3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(nb_classes))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = Getmodel_tensorflow(5)
model.load_weights("./model/plate_type.h5")
model.save("./model/plate_type.h5")
def SimplePredict(image):
    image = cv2.resize(image, (34, 9))
    image = image.astype(np.float) / 255
    res = np.array(model.predict(np.array([image]))[0])
    return res.argmax()


