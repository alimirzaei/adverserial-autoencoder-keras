# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape
from keras.datasets import mnist
from keras.optimizers import sgd
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

img_shape = (28, 28)
encoded_dim = 2

# build encoder

encoder = Sequential()
encoder.add(Flatten(input_shape=img_shape))
encoder.add(Dense(100, activation='relu'))
encoder.add(Dense(100, activation='relu'))
encoder.add(Dense(encoded_dim, activation='relu'))

encoder.summary()

img = Input(shape=img_shape)
encoded_repr = encoder(img)

# Decoder
decoder = Sequential()

decoder.add(Dense(100,activation='relu', input_dim=encoded_dim))
decoder.add(Dense(100,activation='relu'))
decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
decoder.add(Reshape(img_shape))

decoder.summary()

gen_img = decoder(encoded_repr)

generator = Model(img, gen_img)

generator.compile(optimizer='adadelta', loss='binary_crossentropy')

generator.fit(x_train, x_train, epochs=10, batch_size=100)
decoded_imgs = generator.predict(x_test)
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
