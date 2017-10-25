#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:03:44 2017

@author: ali
"""
from keras.layers import Dense, Activation, BatchNormalization, Reshape, Conv2D, UpSampling2D, MaxPooling2D, Input, Flatten
from keras.models import Sequential, Model
from keras.losses import mse, binary_crossentropy
from keras.datasets import mnist, cifar10
from keras.callbacks import ModelCheckpoint
import numpy as np

def decoder_model():
    model = Sequential()
#    model.add(Dense(input_dim=100, output_dim=1024))
#    model.add(Activation('tanh'))
    model.add(Dense(1024*4*4,  input_dim=20))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((4, 4, 1024), input_shape=(1024*4*4,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(512, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

def encoder_model():
    model = Sequential()
    model.add(Conv2D(128, (3,3),padding='same', input_shape=(32,32,3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256,(5,5), padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512,(5,5), padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024,(4,4), padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train.astype(np.float32)-128) / 255.
    x_test = (x_test.astype(np.float32)-128) / 255.

    img = Input(shape=(32,32,3))
    encoder = encoder_model()
    encoder.summary()
    decoder = decoder_model()
    decoder.summary()
    code = encoder(img)
    img_reconst = decoder(code)
    autoencoder = Model(img, img_reconst)
    autoencoder.summary()
#    autoencoder.load_weights('weights.00.hdf5')
    autoencoder.compile(optimizer='sgd',loss='mse')
    autoencoder.fit(x_train, x_train, batch_size=10, epochs=1, callbacks=[ModelCheckpoint('weights.{epoch:02d}.hdf5', 
                                                                       verbose=0, 
                                                                       save_best_only=False, 
                                                                       save_weights_only=False, 
                                                                       mode='auto', 
                                                                       period=1)])
