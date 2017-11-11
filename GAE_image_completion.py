# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# Author : Ali Mirzaei
# Date : 19/09/2017

import glob
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, Lambda
from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
import helpers
from sklearn.model_selection import GridSearchCV
import keras
from keras.initializers import RandomNormal
import keras.backend as K

initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)

def selector(args):
    xs, ys, image = args
    xs = K.cast(xs, K.tf.int32)
    ys = K.cast(ys, K.tf.int32)
    image = Reshape((28,28))(image)
#    x = xy_s[0:100]
#    y = xy_s[100:200]
    img = K.zeros((28,28))
    #tt=image[xs[0,:],ys[0,:]]
    return image

class GAE():
    def __init__(self, img_shape=(28, 28), encoded_dim=2):
        self.encoded_dim = encoded_dim
        self.optimizer = Adam(0.001)
        self.optimizer_discriminator = Adam(0.00001)
        self._initAndCompileFullModel(img_shape, encoded_dim)
        self.img_shape = img_shape

    def _genEncoderModel(self, img_shape, encoded_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        encoder = Sequential()
        encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dense(encoded_dim))
        encoder.summary()
        return encoder

    def _getDecoderModel(self, encoded_dim, img_shape):
        """ Build Decoder Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
            img_shape (tuple) : shape of target images
        Return:
            A sequential keras model
        """
        decoder = Sequential()
        decoder.add(Dense(1000, activation='relu', input_dim=encoded_dim))
        decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        decoder.summary()
        return decoder

    def _getLocationNetwork(self, img_shape):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        locationNetwork = Sequential()
        locationNetwork.add(Flatten(input_shape=img_shape))
        locationNetwork.add(Dense(1000, activation='relu',
                                kernel_initializer=initializer,
                bias_initializer=initializer))
        locationNetwork.add(Dense(1000, activation='relu', kernel_initializer=initializer,
                bias_initializer=initializer))
        locationNetwork.add(Dense(100, activation='sigmoid', kernel_initializer=initializer,
                bias_initializer=initializer))
        locationNetwork.summary()
        img = Input(shape=img_shape)
        xs = locationNetwork(img)
        ys = locationNetwork(img)
        #img_xys = K.concatenate([xys, Flatten()(img)])
        out = Lambda(selector,output_shape=img_shape)([xs, ys,img])
        return Model(img,out)

    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        img = Input(shape=img_shape)
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        self.autoencoder.compile(optimizer=self.optimizer, loss='mse')
        self.locator = self._getLocationNetwork(img_shape)
        self.locator.compile(optimizer=self.optimizer, loss='mse')

    def imagegrid(self, epochnumber):
        fig = plt.figure(figsize=[20, 20])
        for i in range(-5, 5):
            for j in range(-5,5):
                topred = np.array((i*0.5,j*0.5))
                topred = topred.reshape((1, 2))
                img = self.decoder.predict(topred)
                img = img.reshape(self.img_shape)
                ax = fig.add_subplot(10, 10, (i+5)*10+j+5+1)
                ax.set_axis_off()
                ax.imshow(img, cmap="gray")
        fig.savefig(str(epochnumber)+".png")
        plt.show()
        plt.close(fig)

    def train(self, x_in, x_out, batch_size=32, epochs=5):
        self.autoencoder.load_weights('weights.04.hdf5')
        self.autoencoder.fit(x_in, x_out, epochs=epochs, batch_size=batch_size,
                              callbacks=[keras.callbacks.ModelCheckpoint('weights.{epoch:02d}.hdf5', 
                                           verbose=0, 
                                           save_best_only=False, 
                                           save_weights_only=False, 
                                           mode='auto', 
                                           period=1),
    keras.callbacks.TensorBoard(log_dir='./logs', 
                                histogram_freq=0, 
                                batch_size=batch_size, write_graph=True,
                                write_grads=False,
                                write_images=False,
                                embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None)])

    def trainGAN(self, x_train, epochs =1000, batch_size=32):
        half_batch = batch_size/2
        for epoch in range(epochs):
            #---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs_real = x_train[idx]
            # Generate a half batch of new images
            imgs_fake = self.generate(n = half_batch)
            #gen_imgs = self.decoder.predict(latent_fake)
            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(imgs_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            #d_loss = (0,0)
            codes = self.kde.sample(batch_size)
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))
            # Train generator
            g_logg_similarity = self.decoder_discriminator.train_on_batch(codes, valid_y)
            # Plot the progress
            print ("%d [D accuracy: %.2f] [G accuracy: %.2f]" % (epoch, d_loss[1], g_logg_similarity[1]))
#            if(epoch % save_interval == 0):
#                self.imagegrid(epoch)

    def generate(self, n = 10000):
        codes = self.kde.sample(n)
        images = self.decoder.predict(codes)
        return images

    def generateAndPlot(self, x_test, n = 10, fileName="generated.png"):
        fig = plt.figure(figsize=[20, 20*n/3])
        for i in range(n):
            x_in = x_test[np.random.randint(len(x_test))]
            x=copy.copy(x_in)
            mask = np.random.choice([0, 1], size=(28, 28), p=[1./10, 9./10])
            mm = np.ma.masked_array(x, mask= mask)
            x[mm.mask]=0
            y = self.autoencoder.predict(x.reshape(1, 28, 28))
            ax = fig.add_subplot(n, 3, i*3+1)
            ax.set_axis_off()
            ax.imshow(x)
            ax = fig.add_subplot(n, 3, i*3+2)
            ax.set_axis_off()
            ax.imshow(y[0])
            ax = fig.add_subplot(n, 3, i*3+3)
            ax.set_axis_off()
            ax.imshow(x_in)
        fig.savefig(fileName)
        plt.show()

    def meanLogLikelihood(self, x_test):
        KernelDensity(kernel='gaussian', bandwidth=0.2).fit(codes)
import copy
if __name__ == '__main__':
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_out = copy.copy(x_train)
    x_in = []
    for x in x_train:
        mask = np.random.choice([0, 1], size=(28, 28), p=[1./10, 9./10])
        mm = np.ma.masked_array(x, mask= mask)
        x[mm.mask]=0
        x_in.append(x)
    x_in = np.array(x_in)
    ann = GAE(img_shape=(28,28), encoded_dim=8)
    #ann.train(x_in,x_out, epochs=0)
    ann.locator.fit(x_train, x_train)
    ann.generateAndPlot(x_test,50)
#    ann.generateAndPlot(x_train)
#    generated = ann.generate(10000)
#    L = helpers.approximateLogLiklihood(generated, x_test, searchSpace=[.1])
#    print L
#    #codes = ann.kde.sample(1000)
#    #ax = Axes3D(plt.gcf())
#    codes = ann.encoder.predict(x_train)
#    plt.scatter(codes[:,0], codes[:,1], c=y_train)
#    plt.show()
