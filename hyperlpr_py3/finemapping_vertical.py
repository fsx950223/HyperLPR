#coding=utf-8
import numpy as np
import tensorflow as tf

import cv2

def getModel():
    input = tf.keras.layers.Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = tf.keras.layers.Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = tf.keras.layers.Activation("relu", name='relu1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = tf.keras.layers.Activation("relu", name='relu2')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = tf.keras.layers.Activation("relu", name='relu3')(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(2,name = "dense")(x)
    output = tf.keras.layers.Activation("relu", name='relu4')(output)
    model = tf.keras.Model([input], [output])
    return model

model = getModel()
model.load_weights("./model/model12.h5")


def getmodel():
    return model

def gettest_model():
    input = tf.keras.layers.Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    A = tf.keras.layers.Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    B = tf.keras.layers.Activation("relu", name='relu1')(A)
    C = tf.keras.layers.MaxPool2D(pool_size=2)(B)
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(C)
    x = tf.keras.layers.Activation("relu", name='relu2')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    K = tf.keras.layers.Activation("relu", name='relu3')(x)


    x = tf.keras.layers.Flatten()(K)
    dense = tf.keras.layers.Dense(2,name = "dense")(x)
    output = tf.keras.layers.Activation("relu", name='relu4')(dense)
    x = tf.keras.Model([input], [output])
    x.load_weights("./model/model12.h5")
    ok = tf.keras.Model([input], [dense])

    return ok


def finemappingVertical(image):
    resized = cv2.resize(image,(66,16))
    resized = resized.astype(np.float)/255
    res = model.predict(np.array([resized]))[0]
    res = res * image.shape[1]
    res = res.astype(np.int)
    H, T = res
    H -= 3

    if H < 0:
        H = 0
    T += 2;

    if T >= image.shape[1]-1:
        T = image.shape[1]-1

    image = image[0:35, H:T+2]

    image = cv2.resize(image, (int(136), int(36)))
    return image